import cv2
import numpy as np

def is_power_of_two(n):
    
    return n >= 2 and n <= 256 and (n & (n - 1)) == 0

def quantize_image(img, levels):
    step = 256 // levels
    return np.floor(img / step) * step

def add_label_to_image(image, label):
    
    # Adds a label above the image with a white header
    label_height = 40
    header = np.full((label_height, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((header, image))

def resize_to_fit(img, max_height=400):
    
    # Resize image to fit a specific height while keeping aspect ratio
    h, w = img.shape[:2]
    scale = max_height / h
    new_size = (int(w * scale), max_height)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def process_image(path, levels):
    
    if not is_power_of_two(levels):
        
        raise ValueError("Levels must be a power of 2 between 2 and 256.")

    # Load original image in color
    color_img = cv2.imread(path)
    if color_img is None:
        raise FileNotFoundError("Image not found or invalid path.")

    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Reduce intensity levels
    reduced_img = quantize_image(gray_img, levels).astype(np.uint8)

    # Convert grayscale images to BGR for color labeling
    gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    reduced_bgr = cv2.cvtColor(reduced_img, cv2.COLOR_GRAY2BGR)

    # Resize both to fit nicely
    gray_bgr_resized = resize_to_fit(gray_bgr)
    reduced_bgr_resized = resize_to_fit(reduced_bgr)

    # Add headers
    labeled_gray = add_label_to_image(gray_bgr_resized, "Original Grayscale")
    labeled_reduced = add_label_to_image(reduced_bgr_resized, f"Reduced to {levels} Levels")

    # Match height before stacking
    final_height = min(labeled_gray.shape[0], labeled_reduced.shape[0])
    labeled_gray = cv2.resize(labeled_gray, (labeled_gray.shape[1], final_height))
    labeled_reduced = cv2.resize(labeled_reduced, (labeled_reduced.shape[1], final_height))

    # Combine side-by-side
    combined = np.hstack((labeled_gray, labeled_reduced))

    return combined

def main():
    
    try:
        
        path = input("Enter image path: ").strip()
        levels = int(input("Enter intensity levels (power of 2 between 2 and 256): "))

        combined_image = process_image(path, levels)

        # Show the result
        cv2.imshow("Grayscale Intensity Comparison", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
