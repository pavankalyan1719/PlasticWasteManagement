import io
import math
from urllib.parse import quote

import folium
import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from streamlit_searchbox import st_searchbox
from torchvision import models, transforms

from database import add_request, get_requests, get_request_image, update_status

st.set_page_config(page_title="Plastic Waste Management", page_icon="♻️", layout="wide")
st.title("♻️ Deep Learning-Based Waste Management")
st.write("Report waste at a real location and help coordinate community cleanup activities.")

@st.cache_resource
def load_model():
    checkpoint = torch.load("models/plastic_model.pth", map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes

model, classes = load_model()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

category_names = {
    "E": "Electronic Waste",
    "O": "Organic Waste",
    "R": "Recyclable Waste"
}

def search_locations(searchterm):
    if not searchterm or len(searchterm.strip()) < 2:
        return []

    api_key = st.secrets.get("GEOAPIFY_API_KEY", "")
    if not api_key:
        return []

    try:
        response = requests.get(
            "https://api.geoapify.com/v1/geocode/autocomplete",
            params={
                "text": searchterm.strip(),
                "filter": "countrycode:in",
                "format": "json",
                "limit": 8,
                "lang": "en",
                "apiKey": api_key
            },
            timeout=8
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            (
                result.get("formatted", result.get("address_line1", "Unknown location")),
                {
                    "address": result.get("formatted", ""),
                    "lat": result.get("lat"),
                    "lon": result.get("lon")
                }
            )
            for result in results
            if result.get("formatted")
            and result.get("lat") is not None
            and result.get("lon") is not None
        ]
    except requests.RequestException:
        return []

def calculate_distance(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(
        math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(max(0.0, min(1.0, a))))

def make_maps_url(location):
    parts = location.split(" || ")
    coordinates = None

    for prefix in ("PIN:", "GEO:"):
        for part in parts:
            if part.startswith(prefix):
                coordinates = part[len(prefix):]
                break
        if coordinates:
            break

    destination = coordinates or parts[0]
    return (
        "https://www.google.com/maps/dir/?api=1&destination="
        f"{quote(destination)}&travelmode=driving"
    )

def display_location(location):
    return location.split(" || ", 1)[0]

page = st.sidebar.radio("Select Page", ["Report Waste", "Team Dashboard"])

if page == "Report Waste":
    st.header("Submit a Waste Report")

    photo_purpose = st.radio(
        "What is this photograph for?",
        ["Real Cleanup Request", "Demo / Testing Only"],
        horizontal=True,
        key="photo_purpose"
    )

    if photo_purpose == "Real Cleanup Request":
        st.info(
            "Use a photograph of the actual waste at the location you report. "
            "Google or other internet images must only be used for demo/testing."
        )
        real_photo_confirmed = st.checkbox(
            "I confirm this photo shows the actual waste at the selected site.",
            key="real_photo_confirmation"
        )
    else:
        st.warning(
            "Demo images can be classified by the model, but they will NOT be "
            "saved as cleanup requests or shown to the cleanup team."
        )
        real_photo_confirmed = False

    image_method = st.radio(
        "Choose how to provide the photograph",
        ["Take a Photo", "Upload an Image"],
        horizontal=True
    )

    if image_method == "Take a Photo":
        uploaded_file = st.camera_input("Capture a photograph of the waste")
    else:
        uploaded_file = st.file_uploader(
            "Upload a photograph of the waste",
            type=["jpg", "jpeg", "png"]
        )

    st.subheader("📍 Waste Location")
    st.caption("Search for the area, then click the exact waste spot on the map.")

    selected_location = st_searchbox(
        search_function=search_locations,
        placeholder="Search a street, area, city or district...",
        label="Search for a location in India",
        key="waste_location_search",
        debounce=400,
        clear_on_submit=False,
        edit_after_submit="option"
    )

    address = ""
    selected_lat = None
    selected_lon = None
    pin_lat = None
    pin_lon = None
    location = ""

    if isinstance(selected_location, dict):
        address = selected_location.get("address", "")
        selected_lat = selected_location.get("lat")
        selected_lon = selected_location.get("lon")

        if address and selected_lat is not None and selected_lon is not None:
            selected_lat = float(selected_lat)
            selected_lon = float(selected_lon)

            if st.session_state.get("map_address") != address:
                st.session_state["map_address"] = address
                st.session_state["pin_lat"] = selected_lat
                st.session_state["pin_lon"] = selected_lon
                st.session_state.pop("map_last_click", None)

            st.success(f"Selected address: {address}")
            st.info("The marker starts at the searched location. Click the map to move it to the exact waste site.")

            map_lat = float(st.session_state.get("pin_lat", selected_lat))
            map_lon = float(st.session_state.get("pin_lon", selected_lon))

            waste_map = folium.Map(
                location=[map_lat, map_lon],
                zoom_start=17,
                control_scale=True
            )
            folium.Marker(
                [map_lat, map_lon],
                tooltip="Selected waste location",
                popup="Waste pickup point",
                icon=folium.Icon(color="red", icon="trash", prefix="fa")
            ).add_to(waste_map)

            map_result = st_folium(
                waste_map,
                width=None,
                height=450,
                key="waste_site_map",
                returned_objects=["last_clicked"]
            )

            clicked = map_result.get("last_clicked") if map_result else None

            if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
                clicked_lat = float(clicked["lat"])
                clicked_lon = float(clicked["lng"])
                click_key = (round(clicked_lat, 7), round(clicked_lon, 7))

                if click_key != st.session_state.get("map_last_click"):
                    st.session_state["pin_lat"] = clicked_lat
                    st.session_state["pin_lon"] = clicked_lon
                    st.session_state["map_last_click"] = click_key
                    st.rerun()

            pin_lat = float(st.session_state.get("pin_lat", selected_lat))
            pin_lon = float(st.session_state.get("pin_lon", selected_lon))
            location = f"{address} || GEO:{selected_lat},{selected_lon} || PIN:{pin_lat},{pin_lon}"

            st.success("Waste-site pin selected.")
            col1, col2 = st.columns(2)
            col1.metric("Pinned latitude", f"{pin_lat:.6f}")
            col2.metric("Pinned longitude", f"{pin_lon:.6f}")

            st.link_button(
                "📍 Open Pinned Waste Location in Google Maps",
                f"https://www.google.com/maps/search/?api=1&query={pin_lat},{pin_lon}"
            )

    elif isinstance(selected_location, str) and selected_location:
        st.info("Select a location from the autocomplete suggestions.")

    st.subheader("🛰️ Current Device Location (Optional)")
    st.warning(
        "Device GPS shows the device's current position. It does not prove "
        "where an uploaded photograph was taken."
    )

    gps_location = streamlit_geolocation()
    gps_lat = None
    gps_lon = None
    gps_accuracy = None

    if isinstance(gps_location, dict):
        latitude = gps_location.get("latitude")
        longitude = gps_location.get("longitude")
        accuracy = gps_location.get("accuracy")

        if latitude is not None and longitude is not None:
            try:
                gps_lat = float(latitude)
                gps_lon = float(longitude)
                gps_accuracy = float(accuracy) if accuracy is not None else None
                st.success("Device GPS coordinates detected.")

                col1, col2 = st.columns(2)
                col1.metric("GPS latitude", f"{gps_lat:.6f}")
                col2.metric("GPS longitude", f"{gps_lon:.6f}")

                if gps_accuracy is not None:
                    st.caption(f"Reported GPS accuracy: approximately {gps_accuracy:.1f} metres.")

                st.link_button(
                    "🌍 View Current Device GPS",
                    f"https://www.google.com/maps/search/?api=1&query={gps_lat},{gps_lon}"
                )
            except (TypeError, ValueError):
                gps_lat = None
                gps_lon = None

    if gps_lat is None or gps_lon is None:
        st.info("Allow browser location access to detect GPS. You can still select the waste location manually.")

    mismatch_confirmed = True
    distance_km = None

    if location and pin_lat is not None and pin_lon is not None and gps_lat is not None and gps_lon is not None:
        distance_km = calculate_distance(pin_lat, pin_lon, gps_lat, gps_lon)
        st.subheader("Location Check")
        st.write(f"Distance between the selected waste site and device GPS: **{distance_km:.2f} km**")

        if distance_km > 5:
            st.warning("The selected waste site is more than 5 km from your device. Check that the map pin is correct.")
            mismatch_confirmed = st.checkbox(
                "I checked the map pin and confirm the waste-site location is correct.",
                key="confirm_location_mismatch"
            )
        else:
            st.success("The pinned location is within 5 km of the device GPS position.")

    description = st.text_area(
        "Description",
        placeholder="Describe the waste, approximate quantity and nearby landmarks."
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Photograph Provided", width=350)
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_index = probabilities.max(dim=1)

            predicted_class = classes[predicted_index.item()]
            predicted_category = category_names.get(predicted_class, predicted_class)
            confidence_value = confidence.item()

            st.subheader("🧠 Deep Learning Prediction")
            st.write("Predicted category:", predicted_category)
            st.write(f"Model confidence: {confidence_value * 100:.2f}%")
            st.caption(
                "The model predicts electronic, organic or recyclable waste. "
                "It is not a plastic-specific detector, and its prediction does not verify the photo's location."
            )

            if confidence_value < 0.60:
                st.warning("The AI prediction confidence is low. The team should verify the category.")

            if photo_purpose == "Demo / Testing Only":
                st.success("Demo prediction complete. No cleanup request has been saved.")
            elif st.button("Submit Real Cleanup Request", type="primary"):
                if not real_photo_confirmed:
                    st.error("Confirm that the photo shows the actual waste at the selected site.")
                elif not location or pin_lat is None or pin_lon is None:
                    st.error("Please select an address and set the exact waste-site pin on the map.")
                elif distance_km is not None and distance_km > 5 and not mismatch_confirmed:
                    st.error("Please verify and confirm the selected waste location.")
                else:
                    saved_location = (
                        f"{address} || GEO:{selected_lat},{selected_lon} "
                        f"|| PIN:{pin_lat},{pin_lon}"
                    )

                    if gps_lat is not None and gps_lon is not None:
                        saved_location += f" || GPS:{gps_lat},{gps_lon}"

                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format="JPEG")

                    saved_description = (
                        "[Photo source declared by reporter: real waste-site photograph] "
                        + description.strip()
                    )

                    add_request(
                        saved_location,
                        saved_description,
                        predicted_category,
                        confidence_value,
                        image_bytes.getvalue()
                    )

                    # Set the newly created request to verification status.
                    request_rows = get_requests()
                    if request_rows:
                        newest_request_id = request_rows[0][0]
                        update_status(newest_request_id, "Pending Verification")

                    st.success("Real cleanup request submitted for verification.")
                    st.info("The cleanup team must verify the report before approving cleanup.")
                    st.link_button(
                        "🗺️ Open Directions to Waste Location",
                        make_maps_url(saved_location)
                    )

        except Exception as error:
            st.error(f"Unable to process the photograph: {error}")

elif page == "Team Dashboard":
    st.header("Cleanup Team Dashboard")
    st.caption("Verify each report before approving it for cleanup.")
    requests_list = get_requests()

    if not requests_list:
        st.info("No cleanup requests have been submitted yet.")
    else:
        for request in requests_list:
            request_id, location, description, category, confidence, status = request

            with st.expander(f"Request #{request_id} — {status}"):
                st.write("Waste location:", display_location(location))

                for prefix, label in [
                    ("GEO:", "Autocomplete coordinates"),
                    ("PIN:", "Pinned waste-site coordinates"),
                    ("GPS:", "Reporter device GPS coordinates")
                ]:
                    for part in location.split(" || "):
                        if part.startswith(prefix):
                            st.write(f"{label}:", part[len(prefix):])

                if " || GPS:" in location:
                    st.caption("Device GPS is supporting information, not proof of where the photo was taken.")

                st.link_button(
                    "📍 Open Directions to Waste Site",
                    make_maps_url(location),
                    key=f"map_{request_id}"
                )

                st.write("Description:", description or "No description provided")
                st.write("Predicted category:", category)
                st.write(f"Model confidence: {confidence * 100:.2f}%")

                image_bytes = get_request_image(request_id)
                if image_bytes:
                    try:
                        st.image(
                            Image.open(io.BytesIO(image_bytes)),
                            caption=f"Waste Photograph — Request #{request_id}",
                            width=300
                        )
                    except Exception:
                        st.warning("The stored image could not be displayed.")

                statuses = [
                    "Pending Verification",
                    "Approved for Cleanup",
                    "In Progress",
                    "Completed",
                    "Rejected",
                    "Pending"
                ]
                current_index = statuses.index(status) if status in statuses else 0

                new_status = st.selectbox(
                    "Update request status",
                    statuses,
                    index=current_index,
                    key=f"status_{request_id}"
                )

                if st.button("Save Status", key=f"save_{request_id}"):
                    update_status(request_id, new_status)
                    st.success("Request status updated.")
                    st.rerun()
