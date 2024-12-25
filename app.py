import whisper
import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

# Create temporary directory for file uploads
os.makedirs("temp_videos", exist_ok=True)
os.makedirs("output_files", exist_ok=True)

@app.post("/transcribe/")
async def transcribe_video(
    file: UploadFile,
    model: str = Form("base"),
    output_format: str = Form("srt")
):
    # Save the uploaded video locally
    video_path = f"temp_videos/{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Load the Whisper model
    whisper_model = whisper.load_model(model)

    # Perform transcription
    result = whisper_model.transcribe(video_path)

    # Generate output file based on the selected format
    output_path = f"output_files/{Path(video_path).stem}.{output_format}"
    with open(output_path, "w") as f:
        if output_format == "srt":
            for segment in result["segments"]:
                f.write(
                    f"{segment['id'] + 1}\n"
                    f"{segment['start']:0>2}:{segment['start'] % 60:05.2f} --> {segment['end']:0>2}:{segment['end'] % 60:05.2f}\n"
                    f"{segment['text']}\n\n"
                )
        elif output_format == "txt":
            f.write(result["text"])
        elif output_format == "vtt":
            f.write("WEBVTT\n\n")
            for segment in result["segments"]:
                f.write(
                    f"{segment['start']:0>2}:{segment['start'] % 60:05.2f} --> {segment['end']:0>2}:{segment['end'] % 60:05.2f}\n"
                    f"{segment['text']}\n\n"
                )
        else:
            raise ValueError("Unsupported format.")

    return FileResponse(output_path, media_type="application/octet-stream", filename=Path(output_path).name)
