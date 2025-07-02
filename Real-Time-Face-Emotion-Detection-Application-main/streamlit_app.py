# Import required libraries
import numpy as np
import cv2
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the pre-trained model

import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets full path to this script
model_path = os.path.join(base_dir, 'model_78.h5')     # Builds full path to the model

print("Current working directory:", os.getcwd())
print("Does model exist at path?", os.path.exists(model_path))
print("Model path:", model_path)

from tensorflow.keras.models import load_model

classifier = load_model(r"C:\Users\Anish\OneDrive\Desktop\CODING\Face-Det\Real-Time-Face-Emotion-Detection-Application-main\model_78.h5")
# Load weights into the model
# Ensure the weights file is in the same directory as the model
weights_path = os.path.join(base_dir, 'model_weights_78.h5')
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights file not found at {weights_path}")
classifier.load_weights(r"C:\Users\Anish\OneDrive\Desktop\CODING\Face-Det\Real-Time-Face-Emotion-Detection-Application-main\model_weights_78.h5")

# Load face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Error loading Haar Cascade: {e}")

# Define the video transformer
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi, verbose=0)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
                label_position = (x, y - 10)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# Streamlit main function
def main():
    st.title("Real-Time Face Emotion Detection App 😠🤮😨😀😐😔😮")
    menu = ["Home", "Live Face Emotion Detection", "About"]
    choice = st..selectbox("Select Activity", menu)

    st.sidebar.markdown(
        """ Developed by Anish Dhiman  
        [LinkedIn](www.linkedin.com/in/anish-dhiman-837b61313/)""")

    if choice == "Home":
        st.markdown(
            """<div style="background-color:#FC4C02;padding:0.5px">
            <h4 style="color:white;text-align:center;">Start Your Real Time Face Emotion Detection</h4>
            </div><br>""",
            unsafe_allow_html=True
        )

        st.write("""
        * An average person spends hours in front of a screen. 
        * But has your device ever responded to your emotions? 
        * Let's find out...
        
        **Instructions:**
        1. Go to "Live Face Emotion Detection" in the sidebar.
        2. Allow camera access.
        3. Watch your emotion get detected in real-time!
        """)

    elif choice == "Live Face Emotion Detection":
        st.header("Webcam Live Feed")
        st.subheader("Get ready with all the emotions you can express.")
        st.write("1. Click Start to open your camera.")
        st.write("2. This will predict your emotion.")
        st.write("3. Click Stop to end.")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    elif choice == "About":
        st.subheader("About this app")
        st.markdown(
            """<div style="background-color:#36454F;padding:30px">
            <h4 style="color:white;">
            This app uses a Convolutional Neural Network to predict facial emotion.
            Built using TensorFlow + Keras + Streamlit.
            Face detection is powered by OpenCV.
            </h4></div><br>""",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
