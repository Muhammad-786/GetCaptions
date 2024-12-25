import streamlit as st
import whisper
import tempfile
import os
import subprocess

def extract_audio(video_path, audio_path):
    """Extracts audio from a video file."""
    command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{audio_path}\" -y"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def transcribe_audio(model_name, audio_path):
    """Transcribes audio using Whisper."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result['text']

def save_transcription(text, output_format, output_path):
    """Saves the transcription in the specified format."""
    if output_format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    elif output_format == "srt":
        lines = text.split(". ")
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, line in enumerate(lines, 1):
                f.write(f"{idx}\n00:00:00,000 --> 00:00:10,000\n{line.strip()}\n\n")

# Streamlit UI
st.title("AI Video Transcription")
st.write("Upload a video file, select a model, and choose your output format.")

# Video Upload
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv", "avi"])

# Model Selection
model_name = st.selectbox(
    "Select Whisper Model",
    ["tiny", "base", "small", "medium", "large"],
    index=1
)

# Output Format Selection
output_format = st.selectbox(
    "Select Output Format",
    ["txt", "srt"],
    index=0
)

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name

    # Extract audio to a temporary file
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    st.write("Extracting audio...")
    extract_audio(video_path, audio_path)

    # Transcribe audio
    st.write("Transcribing audio with Whisper...")
    transcription = transcribe_audio(model_name, audio_path)

    # Save transcription
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}").name
    save_transcription(transcription, output_format, output_path)

    # Provide download link
    st.write("Transcription complete!")
    with open(output_path, "rb") as f:
        st.download_button(
            label=f"Download Transcription ({output_format.upper()})",
            data=f,
            file_name=f"transcription.{output_format}",
            mime=f"text/{output_format}"
        )

    # Clean up temporary files
    os.remove(video_path)
    os.remove(audio_path)
    os.remove(output_path)
