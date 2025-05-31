from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    provider="fal-ai",
    api_key=os.getenv("HUGGINGFACE_TOKEN"),
)

output = client.automatic_speech_recognition("8975-270782-0114.flac", model="openai/whisper-large-v3")
print(output)