import cv2
import numpy as np

def rotate_image(image, angle):
    
    # Rotate the image around its center without cropping
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the bounding box of the rotated image
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    return cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=(255, 255, 255))

def add_label_to_image(image, label):
    
    # Add a label above the image
    label_height = 40
    header = np.full((label_height, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((header, image))

def resize_to_fit(img, target_height=300):
    
    # Resize image to a target height keeping aspect ratio
    h, w = img.shape[:2]
    scale = target_height / h
    new_size = (int(w * scale), target_height)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def resize_to_size(img, size=(400, 300)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# Main function to handle image processing
def main():
    
    try:
        path = input("Enter image path: ").strip()

        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError("Image not found or path incorrect.")

        rotated_45 = rotate_image(image, 45)
        rotated_90 = rotate_image(image, 90)

        size = (400, 300)  # fixed size for all images (width, height)
        original_resized = resize_to_size(image, size)
        rotated_45_resized = resize_to_size(rotated_45, size)
        rotated_90_resized = resize_to_size(rotated_90, size)

        labeled_original = add_label_to_image(original_resized, "Original")
        labeled_45 = add_label_to_image(rotated_45_resized, "Rotated 45 degrees")
        labeled_90 = add_label_to_image(rotated_90_resized, "Rotated 90 degrees")

        blank = np.full_like(labeled_original, 255)
        top_row = np.hstack((labeled_original, labeled_45))
        bottom_row = np.hstack((labeled_90, blank))
        grid = np.vstack((top_row, bottom_row))

        cv2.imshow("Image Rotation - 45° and 90°", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
