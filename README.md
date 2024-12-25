# GetCaptions

1,Running the Application
Run the backend server:
bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000

2.Open the frontend:
Place the index.html in the same directory as the app.py.
Start a simple HTTP server to serve the HTML file:
bash
python -m http.server 8080

Open http://localhost:8080 in your browser.

3.Transcribe a video:
Upload a video file.
Select the model and format.
The transcription will process, and a download link will appear for the output file.

Notes:
Whisper Models: Larger models like large provide better accuracy but require more memory and processing time.
Output Formats: Add support for additional formats if required.
