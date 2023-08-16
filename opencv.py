# import cv2 as cv
# #for img captue
# img = cv.imread('2.jpeg')
# cv.imshow('sujit',img)
# cv.waitKey(0)

#for video captue
# capture = cv.VideoCapture('2023-08-11 23-33-35.mp4')
# while True:
#     isTrue,frame = capture.read()
#     cv.imshow('video',frame)
#     if cv.waitKey(20) & 0xFF ==ord('a'):
#         break

# capture.release()
# cv.destroyAllWindows()
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('final_model_weights.hdf5')

# Load and preprocess an image using OpenCV
image_path = 'download (3).jpeg'
image = cv2.imread(image_path)
image = cv2.resize(image, (180, 180))  # Resize to match model input size
image = image / 255.0  # Normalize pixel values

# Make a prediction
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_class = np.argmax(prediction)

# Define classes based on your waste types
classes = ['Plastic', 'Metal', 'Paper', 'Glass', 'Organic', 'Other']

# Get the predicted class label
predicted_class_label = classes[predicted_class]

# Display the result
cv2.putText(image, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Waste Classification', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
