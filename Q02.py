import cv2
import numpy as np

def apply_mean_blur(img, kernel_size):
    
    # Apply average (mean) blur
    return cv2.blur(img, (kernel_size, kernel_size))

def add_label_to_image(image, label):
    
    # Add a label above the image
    label_height = 40
    header = np.full((label_height, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((header, image))

def resize_to_fit(img, target_height=300):
    
    # Resize image to target height keeping aspect ratio
    h, w = img.shape[:2]
    scale = target_height / h
    new_size = (int(w * scale), target_height)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def main():
    
    try:
        
        path = input("Enter image path: ").strip()

        original = cv2.imread(path)
        # Check if the image was loaded successfully
        if original is None:
            raise FileNotFoundError("Image not found or path incorrect.")
        
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Apply filters
        blur_3x3 = apply_mean_blur(gray, 3)
        blur_10x10 = apply_mean_blur(gray, 10)
        blur_20x20 = apply_mean_blur(gray, 20)

        # Convert all to BGR for consistent display
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        blur3_bgr = cv2.cvtColor(blur_3x3, cv2.COLOR_GRAY2BGR)
        blur10_bgr = cv2.cvtColor(blur_10x10, cv2.COLOR_GRAY2BGR)
        blur20_bgr = cv2.cvtColor(blur_20x20, cv2.COLOR_GRAY2BGR)

        # Resize for consistent display
        height = 300  # Set target height
        gray_bgr = resize_to_fit(gray_bgr, height)
        blur3_bgr = resize_to_fit(blur3_bgr, height)
        blur10_bgr = resize_to_fit(blur10_bgr, height)
        blur20_bgr = resize_to_fit(blur20_bgr, height)

        # Add labels
        labeled_gray = add_label_to_image(gray_bgr, "Original Grayscale")
        labeled_3x3 = add_label_to_image(blur3_bgr, "3x3 Mean Filter")
        labeled_10x10 = add_label_to_image(blur10_bgr, "10x10 Mean Filter")
        labeled_20x20 = add_label_to_image(blur20_bgr, "20x20 Mean Filter")

        # Arrange in 2x2 grid
        top_row = np.hstack((labeled_gray, labeled_3x3))
        bottom_row = np.hstack((labeled_10x10, labeled_20x20))
        grid = np.vstack((top_row, bottom_row))

        # Show result
        cv2.imshow("Mean Filtering - 2x2 Layout", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
