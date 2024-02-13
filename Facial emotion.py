#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model
emotion_model = load_model('C:\\Users\\navan\\fer2013_mini_XCEPTION.102-0.66.hdf5',compile=False)

# Load the Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotions = ["Fear", "Disgust", "Angry", "Happy", "Sad", "Surprise", "Neutral"]

# Create a Tkinter window
root = tk.Tk()
root.title("Facial Emotion Detection")

# Set the desired width and height for video display
video_width = 800
video_height = 600

# Create a Canvas for displaying video feed
canvas = tk.Canvas(root, width=video_width, height=video_height)
canvas.pack()

# Open the video capture
cap = cv2.VideoCapture(0)

# Function to process each video frame and update the Tkinter window
def process_frame():
    ret, frame = cap.read()

    if ret:
        # Resize the frame to the desired width and height
        frame = cv2.resize(frame, (video_width, video_height))

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces_coordinates = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Get the sub-frame (face)
            face = gray_frame[y:y+h, x:x+w]
            # Resize to match the model's input shape
            face = cv2.resize(face, (64, 64))
            # Normalize pixel values to be between 0 and 1
            face = face / 255.0
            # Expand dimensions to match the model's expected input shape
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            # Predict the emotion
            emotion_label = emotion_model.predict(face)
            emotion = np.argmax(emotion_label)
            emotion_text = emotions[emotion]

            # Draw rectangle around the face and display emotion
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert the frame to RGB for displaying in Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = Image.fromarray(rgb_frame)
        new_photo = ImageTk.PhotoImage(image=rgb_frame)  # Create a new PhotoImage

        # Update the Canvas with the new frame
        canvas.create_image(0, 0, anchor=tk.NW, image=new_photo)
        canvas.photo = new_photo

        # Call the process_frame function after 10 milliseconds
        root.after(10, process_frame)

# Function to release the camera and close the Tkinter window
def on_closing():
    cap.release()
    root.destroy()

# Bind the 'q' key to the on_closing function
root.bind('q', lambda event=None: on_closing())

# Call the process_frame function to start the video processing
process_frame()

# Start the Tkinter main loop
root.mainloop()


# In[ ]:




