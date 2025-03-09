# Facial-analysis
This project utilizes the fer (Facial Expression Recognition) library to detect emotions from an image using deep learning. The model predicts the emotions present on a human face and displays the detected emotion with a bounding box around the face.
**1.Installation** 
Before running the script, install the required dependencies:
pip install fer opencv-python matplotlib
**Usage**
Ensure you have an image file named image.jpg in your working directory or specify a different path.
Run the script to detect emotions from the image.
```python
from fer import FER
import cv2
import matplotlib.pyplot as plt

# Read the input image
img = cv2.imread('/content/image.jpg')

# Initialize the emotion detector
detector = FER()

# Detect emotions in the image
emotion = detector.detect_emotions(img)
print(emotion)

# Extract bounding box coordinates
(x, y, w, h) = emotion[0]["box"]

# Get the top emotion detected
emotion, score = detector.top_emotion(img)

# Draw a rectangle around the detected face
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the detected emotion and probability
text = f"{emotion} ({score:.2f})"
cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the image with annotations
plt.imshow(img)
plt.axis('off')
plt.show()
```

**Explanation**

1.Read the Image: The script loads the image using OpenCV.
2.Initialize FER Detector: The FER() class is used to detect emotions.
3.Detect Emotions: The detect_emotions() method returns a list of detected faces and their associated emotions.
4.Extract Bounding Box: The coordinates for the detected face are retrieved.
5.Get the Top Emotion: The top_emotion() function finds the most prominent emotion in the image.
6.Draw Bounding Box: A green rectangle is drawn around the detected face.
7.Display Emotion Label: The detected emotion and confidence score are displayed on the image.
8.Show the Image: The modified image is displayed using Matplotlib.

**Output**

After running the script, an image will be displayed with a bounding box around the face and the detected emotion labeled.
