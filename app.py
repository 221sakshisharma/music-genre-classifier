from flask import Flask, render_template, request
from pytubefix import YouTube
from pydub import AudioSegment
import os
import tensorflow as tf
import joblib
from prediction_pipeline import predict_genre

app = Flask(__name__)

# Load your model and scaler once
model = tf.keras.models.load_model("genre_classifier_model.h5")
scaler = joblib.load("std_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        url = request.form["youtube_url"]

        try:
            # Download audio
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            mp4_path = audio_stream.download(filename="temp_audio.mp4")

            # Convert to WAV
            wav_path = "converted_audio.wav"
            audio = AudioSegment.from_file(mp4_path, format="mp4")
            audio.export(wav_path, format="wav")
            os.remove(mp4_path)

            # Predict
            prediction = predict_genre(model, scaler, wav_path).capitalize()
            os.remove(wav_path)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
