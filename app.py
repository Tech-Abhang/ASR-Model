from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os

from modelLoader import load_asr_model

app = FastAPI(title="ASR Transcription Service")

# Load the model once at startup
asr_model = load_asr_model()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Transcribe using model
        result = asr_model(temp_path)

        # Cleanup
        os.remove(temp_path)

        return JSONResponse(content={"transcription": result["text"]})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)