import pandas as pd
import os
import shutil


def organize_images_by_class():
    # Read the CSV file
    df = pd.read_csv('images/ISIC_2017/data.csv')  # Replace with your CSV filename

    # Base directory where images are currently stored
    base_src_dir = 'images/ISIC_2017/clean'

    # Create a directory for organized images
    base_dest_dir = 'images/ISIC_2017/organized'
    os.makedirs(base_dest_dir, exist_ok=True)

    # Create subdirectories for each unique class
    unique_classes = df['class'].unique()
    for class_name in unique_classes:
        class_dir = os.path.join(base_dest_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    # Move each image to its corresponding class directory
    for _, row in df.iterrows():
        image_filename = os.path.basename(row['image_path'])
        src_path = os.path.join(base_src_dir, image_filename)
        dest_path = os.path.join(base_dest_dir, row['class'], image_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Moved {image_filename} to {row['class']} directory")
        else:
            print(f"Warning: {image_filename} not found in source directory")


if __name__ == "__main__":
    organize_images_by_class()