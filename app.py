import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import tf_keras.src.saving.legacy.save

# import tensorflow.python.keras.saving.save

# Load trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')
IMG_SIZE = (128, 128)


# Function to open file explorer and get image path
def load_image():
    root = Tk()
    root.withdraw()  # Hide root window
    root.attributes('-topmost', True)  # Keep file dialog on top
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    root.destroy()  # Destroy root window after selection
    return file_path


# Load image
image_path = load_image()
if image_path:
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, IMG_SIZE)
    image_array = np.expand_dims(image_resized, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(image_array)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "No Tumor"
    accuracy = max(prediction, 1 - prediction) * 100

    # Convert to grayscale and apply thresholding for highlighting tumor
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Show results
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{label} ({accuracy:.2f}% Confidence)")
    plt.axis('off')
    plt.show()
else:
    print("No image selected.")
