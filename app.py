import whisper
import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

app = FastAPI()

# Create directories for temporary files and outputs
os.makedirs("temp_videos", exist_ok=True)
os.makedirs("output_files", exist_ok=True)

@app.post("/transcribe/")
async def transcribe_video(
    file: UploadFile,
    model: str = Form("base"),
    output_format: str = Form("srt")
):
    # Validate model and output format
    valid_models = ["tiny", "base", "small", "medium", "large"]
    valid_formats = ["srt", "txt", "vtt"]
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Supported models: {valid_models}")
    if output_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Invalid output format '{output_format}'. Supported formats: {valid_formats}")

    # Save the uploaded file locally
    try:
        video_path = f"temp_videos/{file.filename}"
        with open(video_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file. Error: {str(e)}")

    # Load Whisper model
    try:
        whisper_model = whisper.load_model(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Whisper model '{model}'. Error: {str(e)}")

    # Perform transcription
    try:
        result = whisper_model.transcribe(video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

    # Generate output file based on the selected format
    try:
        output_path = f"output_files/{Path(video_path).stem}.{output_format}"
        with open(output_path, "w") as f:
            if output_format == "srt":
                for segment in result["segments"]:
                    start_time = format_time(segment["start"])
                    end_time = format_time(segment["end"])
                    f.write(
                        f"{segment['id'] + 1}\n"
                        f"{start_time} --> {end_time}\n"
                        f"{segment['text']}\n\n"
                    )
            elif output_format == "txt":
                f.write(result["text"])
            elif output_format == "vtt":
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    start_time = format_time(segment["start"])
                    end_time = format_time(segment["end"])
                    f.write(
                        f"{start_time} --> {end_time}\n"
                        f"{segment['text']}\n\n"
                    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during output file generation: {str(e)}")

    # Return the generated file as a response
    try:
        return FileResponse(output_path, media_type="application/octet-stream", filename=Path(output_path).name)
    finally:
        # Clean up temporary video file
        if os.path.exists(video_path):
            os.remove(video_path)

def format_time(seconds: float) -> str:
    """
    Helper function to format time in seconds to HH:MM:SS,mmm format for SRT/VTT.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace(".", ",")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors.
    """
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )
