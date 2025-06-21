import cv2
import numpy as np

def block_average_downsample(img, block_size):
    
    """
    Replace each non-overlapping block_size x block_size block with its average.
    Works for grayscale images.
    """
    h, w = img.shape
    
    # Crop image to make dimensions multiples of block_size
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    img_cropped = img[:h_crop, :w_crop].copy()

    for row in range(0, h_crop, block_size):
        for col in range(0, w_crop, block_size):
            block = img_cropped[row:row+block_size, col:col+block_size]
            avg_val = int(np.mean(block))
            img_cropped[row:row+block_size, col:col+block_size] = avg_val

    return img_cropped

def add_label(image, label):
    
    # Add a white label bar with text above the image
    label_height = 40
    width = image.shape[1]
    header = np.full((label_height, width, 3), 255, dtype=np.uint8)
    cv2.putText(header, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((header, image))

def resize_to_fixed_size(img, size=(400, 300)):
    
    # Resize image to fixed size (width, height)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# Main function to handle image processing
def main():
    
    try:
        path = input("Enter image path: ").strip()
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError("Image not found or path incorrect.")

        # Block average downsampling
        avg_3 = block_average_downsample(original, 3)
        avg_5 = block_average_downsample(original, 5)
        avg_7 = block_average_downsample(original, 7)

        # Convert grayscale images to BGR for display and labeling
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        avg3_bgr = cv2.cvtColor(avg_3, cv2.COLOR_GRAY2BGR)
        avg5_bgr = cv2.cvtColor(avg_5, cv2.COLOR_GRAY2BGR)
        avg7_bgr = cv2.cvtColor(avg_7, cv2.COLOR_GRAY2BGR)

        # Resize all images to the same fixed size
        fixed_size = (400, 300)  # width x height
        orig_bgr = resize_to_fixed_size(orig_bgr, fixed_size)
        avg3_bgr = resize_to_fixed_size(avg3_bgr, fixed_size)
        avg5_bgr = resize_to_fixed_size(avg5_bgr, fixed_size)
        avg7_bgr = resize_to_fixed_size(avg7_bgr, fixed_size)

        # Add labels above images
        labeled_orig = add_label(orig_bgr, "Original Grayscale")
        labeled_3 = add_label(avg3_bgr, "Block Avg 3x3")
        labeled_5 = add_label(avg5_bgr, "Block Avg 5x5")
        labeled_7 = add_label(avg7_bgr, "Block Avg 7x7")

        # Create 2x2 grid
        top_row = np.hstack((labeled_orig, labeled_3))
        bottom_row = np.hstack((labeled_5, labeled_7))
        grid = np.vstack((top_row, bottom_row))

        # Show the final result
        cv2.imshow("Block Average Downsampling (2x2 grid)", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
