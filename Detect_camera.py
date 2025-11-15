import cv2
import tensorflow as tf
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model("Model/ewaste_model.h5")

# Constants
IMG_SIZE = (128, 128)
CLASS_NAMES = ['E-WASTE', 'NON-E-WASTE']

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0), img  # also return original

def classify_image(image_path):
    input_image, original_image = preprocess_image(image_path)
    prediction = model.predict(input_image)[0][0]

    if prediction > 0.5:
        label = "NON-E-WASTE"
        confidence = prediction
        color = (0, 255, 0)
    else:
        label = "E-WASTE"
        confidence = 1 - prediction
        color = (0, 0, 255)

    print(f"\nPredicted Class: {label}")
    print(f"Surity: {confidence * 100:.2f}%")

    # Show image with result
    display_image = cv2.resize(original_image, (512, 512))
    cv2.putText(display_image, f"{label} ({confidence * 100:.2f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Classification Result", display_image)
    print("[INFO] Close the window to finish.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Paste your image path below (drag and drop works too):")
    image_path = input(">> ").strip().strip('"')  # remove quotes if dropped

    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    classify_image(image_path)

if __name__ == "__main__":
    main()

