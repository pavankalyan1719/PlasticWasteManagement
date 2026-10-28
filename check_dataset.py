import os

folders = {
    "Training": "train",
    "Testing": "test"
}

for dataset_name, folder_path in folders.items():
    print(f"\n{dataset_name} Dataset")

    for category in ["E", "O", "R"]:
        category_path = os.path.join(folder_path, category)

        if os.path.exists(category_path):
            images = [
                file for file in os.listdir(category_path)
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            print(f"{category}: {len(images)} images")
        else:
            print(f"{category}: Folder not found")