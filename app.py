import io
import math
import hashlib
import hmac
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

from database import (
    add_request,
    get_requests,
    get_request_image,
    update_status,
    add_cleanup_team,
    get_cleanup_teams,
    set_team_active,
    assign_cleanup_team,
    get_assignment_details,
    update_assignment_status
)

# Configure the application page.
st.set_page_config(page_title="Plastic Waste Management", page_icon="♻️", layout="wide")
st.title("♻️ Deep Learning-Based Waste Management")
st.write("Report waste at a real location and help coordinate community cleanup activities.")

# Load the trained deep-learning model.
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

# Verify the administrator password using the stored password hash.
def verify_admin_password(password):
    try:
        salt = st.secrets["ADMIN_SALT"]
        expected_hash = st.secrets["ADMIN_PASSWORD_HASH"]
        entered_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200000).hex()
        return hmac.compare_digest(entered_hash, expected_hash)
    except Exception:
        return False

# Transform images into the format expected by the trained model.
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Convert model class labels into readable category names.
category_names = {
    "E": "Electronic Waste",
    "O": "Organic Waste",
    "R": "Recyclable Waste"
}

# Search for Indian locations using the Geoapify autocomplete service.
def search_locations(searchterm):
    if not searchterm or len(searchterm.strip()) < 2:
        return []
    try:
        api_key = st.secrets.get("GEOAPIFY_API_KEY", "")
    except Exception:
        api_key = ""
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
            if result.get("formatted") and result.get("lat") is not None and result.get("lon") is not None
        ]
    except requests.RequestException:
        return []

# Calculate the straight-line distance between two geographic coordinates.
def calculate_distance(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(max(0.0, min(1.0, a))))

# Extract the exact waste-site coordinates from a saved report.
def get_site_coordinates(location):
    for part in location.split(" || "):
        if part.startswith("PIN:"):
            try:
                lat, lon = part.replace("PIN:", "", 1).split(",")
                return float(lat), float(lon)
            except (ValueError, TypeError):
                return None, None
    return None, None

# Estimate round-trip distance and fuel cost from a cleanup team's base.
def estimate_travel(team, site_lat, site_lon):
    team_id, name, area, base_lat, base_lon, mileage, fuel_price, active = team
    if base_lat is None or base_lon is None or mileage is None or mileage <= 0 or fuel_price is None:
        return None, None
    straight_distance = calculate_distance(base_lat, base_lon, site_lat, site_lon)
    estimated_round_trip_km = straight_distance * 1.3 * 2
    estimated_fuel_cost = (estimated_round_trip_km / mileage) * fuel_price
    return round(estimated_round_trip_km, 2), round(estimated_fuel_cost, 2)

# Create a Google Maps driving-directions link for a waste report.
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
    return f"https://www.google.com/maps/dir/?api=1&destination={quote(destination)}&travelmode=driving"

# Show only the readable address from the saved location.
def display_location(location):
    return location.split(" || ", 1)[0]

# Display the application navigation menu.
page = st.sidebar.radio("Select Page", ["Report Waste", "Team Dashboard"])

# Display the waste-report submission page.
if page == "Report Waste":
    st.header("📝 Submit a Waste Report")
    photo_purpose = st.radio("What is this photograph for?", ["Real Cleanup Request", "Demo / Testing Only"], horizontal=True, key="photo_purpose")

    if photo_purpose == "Real Cleanup Request":
        st.info("Use a photograph of the actual waste at the location you report. Internet images must only be used for demo/testing.")
        real_photo_confirmed = st.checkbox("I confirm this photo shows the actual waste at the selected site.", key="real_photo_confirmation")
    else:
        st.warning("Demo images can be classified by the model, but they will NOT be saved as cleanup requests or shown to the cleanup team.")
        real_photo_confirmed = False

    # Let the user capture a photograph or upload an existing image.
    image_method = st.radio("Choose how to provide the photograph", ["Take a Photo", "Upload an Image"], horizontal=True)

    if image_method == "Take a Photo":
        uploaded_file = st.camera_input("Capture a photograph of the waste")
    else:
        uploaded_file = st.file_uploader("Upload a photograph of the waste", type=["jpg", "jpeg", "png"])

    # Allow the user to search for the waste-site address.
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

    # Display an interactive map when a suggested address is selected.
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
            waste_map = folium.Map(location=[map_lat, map_lon], zoom_start=17, control_scale=True)

            folium.Marker(
                [map_lat, map_lon],
                tooltip="Selected waste location",
                popup="Waste pickup point",
                icon=folium.Icon(color="red", icon="trash", prefix="fa")
            ).add_to(waste_map)

            map_result = st_folium(waste_map, width=None, height=450, key="waste_site_map", returned_objects=["last_clicked"])
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

    # Read the device's current GPS coordinates when permission is granted.
    st.subheader("🛰️ Current Device Location (Optional)")
    st.warning("Device GPS shows the device's current position. It does not prove where an uploaded photograph was taken.")
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

    # Compare the selected waste site with the current device GPS.
    mismatch_confirmed = True
    distance_km = None

    if location and pin_lat is not None and pin_lon is not None and gps_lat is not None and gps_lon is not None:
        distance_km = calculate_distance(pin_lat, pin_lon, gps_lat, gps_lon)
        st.subheader("Location Check")
        st.write(f"Distance between the selected waste site and device GPS: **{distance_km:.2f} km**")

        if distance_km > 5:
            st.warning("The selected waste site is more than 5 km from your device. Check that the map pin is correct.")
            mismatch_confirmed = st.checkbox("I checked the map pin and confirm the waste-site location is correct.", key="confirm_location_mismatch")
        else:
            st.success("The pinned location is within 5 km of the device GPS position.")

    # Collect details about the waste being reported.
    description = st.text_area("Description", placeholder="Describe the waste, approximate quantity and nearby landmarks.")

    # Predict the waste category when a photograph is provided.
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
            st.caption("The model predicts electronic, organic or recyclable waste. It is not a plastic-specific detector, and its prediction does not verify the photograph's location.")

            if confidence_value < 0.60:
                st.warning("The AI prediction confidence is low. The team should verify the category.")

            # Do not save demo images as real cleanup requests.
            if photo_purpose == "Demo / Testing Only":
                st.success("Demo prediction complete. No cleanup request has been saved.")

            # Validate and save a genuine cleanup report.
            elif st.button("Submit Real Cleanup Request", type="primary"):
                if not real_photo_confirmed:
                    st.error("Confirm that the photo shows the actual waste at the selected site.")
                elif not location or pin_lat is None or pin_lon is None:
                    st.error("Please select an address and set the exact waste-site pin on the map.")
                elif distance_km is not None and distance_km > 5 and not mismatch_confirmed:
                    st.error("Please verify and confirm the selected waste location.")
                else:
                    saved_location = f"{address} || GEO:{selected_lat},{selected_lon} || PIN:{pin_lat},{pin_lon}"

                    if gps_lat is not None and gps_lon is not None:
                        saved_location += f" || GPS:{gps_lat},{gps_lon}"

                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format="JPEG")
                    saved_description = "[Photo source declared by reporter: real waste-site photograph] " + description.strip()

                    add_request(saved_location, saved_description, predicted_category, confidence_value, image_bytes.getvalue())
                    request_rows = get_requests()

                    if request_rows:
                        update_status(request_rows[0][0], "Pending Verification")

                    st.success("Real cleanup request submitted for verification.")
                    st.info("The cleanup team must verify the report before approving cleanup.")
                    st.link_button("🗺️ Open Directions to Waste Location", make_maps_url(saved_location))

        except Exception as error:
            st.error(f"Unable to process the photograph: {error}")

# Display the password-protected cleanup management dashboard.
elif page == "Team Dashboard":
    st.header("🔐 Cleanup Team Dashboard")

    # Initialize the administrator login state.
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    # Ask for administrator credentials before displaying private dashboard data.
    if not st.session_state["admin_authenticated"]:
        st.info("Log in with your administrator credentials to view cleanup requests.")

        with st.form("admin_login_form"):
            username = st.text_input("Admin Username")
            password = st.text_input("Admin Password", type="password")
            login_clicked = st.form_submit_button("Login", type="primary")

        if login_clicked:
            try:
                saved_username = st.secrets["ADMIN_USERNAME"]
                username_matches = hmac.compare_digest(username.encode("utf-8"), saved_username.encode("utf-8"))

                if username_matches and verify_admin_password(password):
                    st.session_state["admin_authenticated"] = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
            except Exception:
                st.error("Admin credentials are not configured. Run python setup_admin.py in your project terminal first.")

    # Show team management and requests only after successful login.
    else:
        header_col, logout_col = st.columns([5, 1])

        with header_col:
            st.subheader("📊 Cleanup Management")

        with logout_col:
            if st.button("Logout"):
                st.session_state["admin_authenticated"] = False
                st.rerun()

        # Allow administrators to register teams and their travel information.
        st.divider()
        st.subheader("🚛 Cleanup Team Management")

        with st.expander("Register a Cleanup Team", expanded=False):
            with st.form("add_cleanup_team_form"):
                team_name = st.text_input("Team Name")
                service_area = st.text_input("Service Area", placeholder="Example: Guntur")
                base_lat = st.number_input("Team Base Latitude", value=0.0, format="%.6f")
                base_lon = st.number_input("Team Base Longitude", value=0.0, format="%.6f")
                mileage = st.number_input("Vehicle Mileage (km/litre)", min_value=1.0, value=15.0)
                fuel_price = st.number_input("Fuel Price per Litre (₹)", min_value=0.0, value=100.0)
                add_team_clicked = st.form_submit_button("Register Team")

            if add_team_clicked:
                if not team_name.strip() or not service_area.strip():
                    st.error("Enter both the team name and service area.")
                elif not -90 <= base_lat <= 90 or not -180 <= base_lon <= 180:
                    st.error("Enter valid latitude and longitude.")
                elif base_lat == 0.0 and base_lon == 0.0:
                    st.error("Enter the team's actual base coordinates.")
                else:
                    try:
                        add_cleanup_team(team_name, service_area, base_lat, base_lon, mileage, fuel_price)
                        st.success("Cleanup team registered successfully.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Unable to register team: {error}")

        # Display registered teams and allow administrators to activate or deactivate them.
        teams = get_cleanup_teams(active_only=False)

        if teams:
            st.write("Registered Cleanup Teams")

            for team in teams:
                team_id, name, area, team_lat, team_lon, team_mileage, team_fuel, active = team
                team_col1, team_col2, team_col3 = st.columns([3, 2, 2])
                team_col1.write(f"**{name}**")
                team_col2.write(f"Service area: {area}")
                team_col3.write("Active" if active else "Inactive")

                if active:
                    if team_col3.button("Deactivate", key=f"deactivate_team_{team_id}"):
                        set_team_active(team_id, False)
                        st.rerun()
                else:
                    if team_col3.button("Reactivate", key=f"reactivate_team_{team_id}"):
                        set_team_active(team_id, True)
                        st.rerun()
        else:
            st.info("No cleanup teams registered yet. Register real teams before assigning requests.")

        st.divider()

        # Retrieve all submitted cleanup requests.
        requests_list = get_requests()

        # Calculate the request statistics.
        total_requests = len(requests_list)
        pending_verification = sum(1 for request in requests_list if request[5] == "Pending Verification")
        approved_requests = sum(1 for request in requests_list if request[5] == "Approved for Cleanup")
        in_progress_requests = sum(1 for request in requests_list if request[5] == "In Progress")
        completed_requests = sum(1 for request in requests_list if request[5] == "Completed")
        rejected_requests = sum(1 for request in requests_list if request[5] == "Rejected")

        # Display request statistics in the dashboard.
        metric_cols = st.columns(3)
        metric_cols[0].metric("Total Requests", total_requests)
        metric_cols[1].metric("Pending Verification", pending_verification)
        metric_cols[2].metric("Approved", approved_requests)

        metric_cols = st.columns(3)
        metric_cols[0].metric("In Progress", in_progress_requests)
        metric_cols[1].metric("Completed", completed_requests)
        metric_cols[2].metric("Rejected", rejected_requests)

        st.divider()
        st.subheader("📋 Cleanup Requests")
        st.caption("Review the photograph and location carefully before approving a cleanup.")

        # Display every submitted report with its image and management controls.
        if not requests_list:
            st.info("No cleanup requests have been submitted yet.")
        else:
            for request in requests_list:
                request_id, location, description, category, confidence, status = request

                with st.expander(f"Request #{request_id} — {status}"):
                    st.write("Waste location:", display_location(location))

                    # Display the saved address, pin, and optional GPS coordinates.
                    for prefix, label in [
                        ("GEO:", "Autocomplete coordinates"),
                        ("PIN:", "Pinned waste-site coordinates"),
                        ("GPS:", "Reporter device GPS coordinates")
                    ]:
                        for part in location.split(" || "):
                            if part.startswith(prefix):
                                st.write(f"{label}:", part[len(prefix):])

                    if " || GPS:" in location:
                        st.caption("Device GPS is supporting information; it does not prove where the photograph was taken.")

                    # Provide a Google Maps route to the reported waste site.
                    st.link_button("📍 Open Directions to Waste Site", make_maps_url(location), key=f"map_{request_id}")

                    # Display the report description and model prediction.
                    st.write("Description:", description or "No description provided")
                    st.write("Predicted category:", category)
                    st.write(f"Model confidence: {confidence * 100:.2f}%")

                    # Retrieve and display the stored photograph.
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

                    # Allow the administrator to update the cleanup request status.
                    statuses = [
                        "Pending Verification",
                        "Approved for Cleanup",
                        "In Progress",
                        "Completed",
                        "Rejected",
                        "Pending"
                    ]

                    current_index = statuses.index(status) if status in statuses else 0
                    new_status = st.selectbox("Update request status", statuses, index=current_index, key=f"status_{request_id}")

                    if st.button("Save Status", key=f"save_{request_id}"):
                        update_status(request_id, new_status)
                        st.success("Request status updated.")
                        st.rerun()

                    # Recommend nearby registered teams and allow assignment decisions.
                    st.divider()
                    st.subheader("🚛 Team Assignment")

                    site_lat, site_lon = get_site_coordinates(location)
                    active_teams = get_cleanup_teams(active_only=True)
                    assignment = get_assignment_details(request_id)

                    if assignment:
                        assigned_team_name = assignment[2]
                        saved_distance = assignment[4]
                        saved_cost = assignment[5]
                        assignment_status = assignment[6]

                        if assigned_team_name:
                            st.write(f"**Assigned Team:** {assigned_team_name}")
                            st.write(f"**Estimated Round-Trip Distance:** {saved_distance} km")
                            st.write(f"**Estimated Round-Trip Fuel Cost:** ₹{saved_cost:.2f}")
                            st.write(f"**Assignment Status:** {assignment_status}")
                        else:
                            st.info(f"Assignment status: {assignment_status}")

                    # Check that the report has usable coordinates and an active team exists.
                    if not active_teams:
                        st.warning("No active cleanup teams are registered.")
                    elif site_lat is None or site_lon is None:
                        st.warning("The waste-site coordinates are unavailable. Verify the report location before assigning a team.")
                    else:
                        ranked_teams = []

                        # Estimate the travel distance and cost for every active team.
                        for team in active_teams:
                            round_trip_km, estimated_cost = estimate_travel(team, site_lat, site_lon)

                            if round_trip_km is not None and estimated_cost is not None:
                                straight_distance = calculate_distance(team[3], team[4], site_lat, site_lon)
                                ranked_teams.append((straight_distance, team, round_trip_km, estimated_cost))

                        ranked_teams.sort(key=lambda item: item[0])

                        if ranked_teams:
                            st.caption("Teams are ranked by straight-line distance. Road distance uses a rough 1.3 multiplier; actual routes and fuel costs may differ.")

                            # Show the nearest teams and their estimated travel costs.
                            for straight_distance, team, round_trip_km, estimated_cost in ranked_teams:
                                st.write(
                                    f"**{team[1]}** — Service area: {team[2]} — "
                                    f"Straight-line distance: {straight_distance:.2f} km — "
                                    f"Estimated round-trip distance: {round_trip_km:.2f} km — "
                                    f"Estimated fuel: ₹{estimated_cost:.2f}"
                                )

                            # Let the administrator choose which team should receive the request.
                            team_options = {
                                f"{team[1]} | {team[2]} | estimated fuel ₹{cost:.2f}": (team, round_trip_km, cost)
                                for straight_distance, team, round_trip_km, cost in ranked_teams
                            }

                            selected_team_label = st.selectbox(
                                "Choose a cleanup team",
                                list(team_options.keys()),
                                key=f"assign_team_select_{request_id}"
                            )

                            assign_col, pending_col = st.columns(2)

                            # Save the chosen team and estimated travel details.
                            if assign_col.button("Assign Selected Team", key=f"assign_team_{request_id}"):
                                team, round_trip_km, estimated_cost = team_options[selected_team_label]

                                try:
                                    assign_cleanup_team(request_id, team[0], round_trip_km, estimated_cost)
                                    update_assignment_status(request_id, "Assigned")
                                    st.success(f"Request assigned to {team[1]}.")
                                    st.rerun()
                                except Exception as error:
                                    st.error(f"Unable to assign team: {error}")

                            # Leave the request unassigned when the available teams are unsuitable.
                            if pending_col.button("Leave Assignment Pending", key=f"pending_assignment_{request_id}"):
                                update_assignment_status(request_id, "Pending Assignment")
                                st.info("Request left pending for a later assignment decision.")
                                st.rerun()

                            st.caption("Choose a team based on real availability and service area. The nearest team is only a recommendation, not a guarantee of availability.")
                        else:
                            st.warning("No teams have valid base coordinates and mileage information.")