from flask import Flask, render_template, request
from pydub import AudioSegment
import os
import tensorflow as tf
import joblib
import yt_dlp
from prediction_pipeline import predict_genre

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("genre_classifier_model.h5")
scaler = joblib.load("std_scaler.pkl")

# Function to download YouTube audio
def download_youtube_audio(url, output_path="temp_audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        url = request.form["youtube_url"]

        try:
            # Step 1: Download
            mp3_path = download_youtube_audio(url)

            # Step 2: Convert to WAV
            wav_path = "converted_audio.wav"
            audio = AudioSegment.from_file(mp3_path, format="mp3")
            audio.export(wav_path, format="wav")
            os.remove(mp3_path)

            # Step 3: Predict
            prediction = predict_genre(model, scaler, wav_path).capitalize()

            # Clean up
            os.remove(wav_path)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
