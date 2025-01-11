import cv2
from pathlib import Path
from segmentation_v2 import ImageProcessing

# Define paths
base_dir = Path("images/ISIC_2017")
train_clean_dir = base_dir / "organized"
segmented_dir = base_dir / "masks"
output_dir = base_dir / "masked_lesions"


def apply_mask_and_save(image_path, segmented_dir, output_dir):
    try:
        # Get image filename and create paths
        image_name = image_path.name
        segmentation_name = image_name.replace('.jpg', '_Segmentation.png')
        segmentation_path = segmented_dir / segmentation_name

        # Read original image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error reading image: {image_path}")
            return

        # Create output directory if it doesn't exist
        output_subfolder = output_dir / image_path.parent.name
        output_subfolder.mkdir(parents=True, exist_ok=True)
        output_path = output_subfolder / image_name

        if segmentation_path.exists():
            # Read and apply existing segmentation mask
            segmented = cv2.imread(str(segmentation_path))
            if segmented is None:
                print(f"Error reading segmentation mask: {segmentation_path}")
                return

            # Apply mask
            mask_applied = cv2.bitwise_and(image, segmented)
            cv2.imwrite(str(output_path), mask_applied)
            print(f"Processed: {image_name}")
        else:
            # Generate and apply new mask
            #img_processor = ImageProcessing(image)
            #mask = img_processor.global_thresholding()

            # Convert mask to match format of existing masks
            # If original masks are 3-channel (BGR)
            #mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Apply mask to original image
            #masked_image = cv2.bitwise_and(image, cv2.bitwise_not(mask_3channel))
            #cv2.imwrite(str(output_path), masked_image)
            print(f"Generated new mask for: {image_name}")

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


def main():
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all subfolders in Test_Clean
    for subfolder in train_clean_dir.iterdir():
        if subfolder.is_dir():
            print(f"\nProcessing folder: {subfolder.name}")

            # Process all images in subfolder
            for image_path in subfolder.glob('*.jpg'):
                apply_mask_and_save(image_path, segmented_dir, output_dir)


if __name__ == "__main__":
    main()