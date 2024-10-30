from flask import Flask, render_template, request
import os, cv2, pandas as pd
from deepface import DeepFace
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from textblob import TextBlob
from collections import defaultdict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['STATIC_FOLDER'] = './static/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure folder exists

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Get patient name and video file from the form
    patient_name = request.form['patient_name']
    video = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    # Process the video and get analysis results
    result = process_video(video_path, patient_name)
    return render_template('index.html', result=result)  # Pass result to the template

def process_video(video_path, patient_name):
    # Initialize variables
    cap = cv2.VideoCapture(video_path)
    emotion_counts = defaultdict(int)
    recognizer = sr.Recognizer()
    overall_sentiment = {'polarity': 0, 'subjectivity': 0}
    total_frames_analyzed = 0
    frame_skip = 10
    image_path = os.path.join(app.config['STATIC_FOLDER'], 'patient_image.jpg')    

    # Extract the first frame with a face and save it
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 1:  # If a face is detected
            cv2.imwrite(image_path, frame)  # Save the frame as an image
            break  # Exit after saving the first face frame

    cap.release()

    # Extract audio and perform sentiment analysis
    video = VideoFileClip(video_path)
    audio = video.audio
    if audio:
        audio.write_audiofile("audio.wav")
        with sr.AudioFile("audio.wav") as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                sentiment = TextBlob(text).sentiment
                overall_sentiment['polarity'] = sentiment.polarity
                overall_sentiment['subjectivity'] = sentiment.subjectivity
            except:
                print("Error processing audio.")

    # Perform emotion detection on video frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion_counts[dominant_emotion] += 1
                total_frames_analyzed += 1
            except:
                print(f"Error processing frame {frame_count}")

        frame_count += 1

    cap.release()

    # Calculate emotion percentages
    emotion_percentages = {
        emotion: (count / total_frames_analyzed) * 100 for emotion, count in emotion_counts.items()
    }
    dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)

    # Combine all results into a dictionary
    result = {
        'patient_name': patient_name,
        'image_path': image_path,
        'dominant_emotion': dominant_emotion,
        'emotion_percentages': emotion_percentages,
        'overall_sentiment': overall_sentiment
    }
    return result

if __name__ == '__main__':
    app.run(debug=True)
