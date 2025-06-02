"""
Client for making inference requests to local or hosted Hugging Face models.
"""
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
import numpy as np

class InferenceModelClient:
    """
    Client for running inference on audio files using Whisper models.
    Supports both local models and Hugging Face hosted models.
    """
    def __init__(self, model_path=None, model_id="openai/whisper-small"):
        """
        Initialize the inference client.
        
        Args:
            model_path (str): Path to local model directory. If None, uses the model_id.
            model_id (str): Hugging Face model ID to use if model_path is None.
        """
        self.model_path = model_path
        self.model_id = model_id if model_path is None else model_path
        self.model = None
        self.processor = None
        
        # Load the model and processor
        self.load_model()
    
    def load_model(self):
        """
        Load the Whisper model and processor.
        """
        print(f"Loading model from {'local path: ' + self.model_path if self.model_path else 'Hugging Face: ' + self.model_id}...")
        
        try:
            # Load processor and model
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
            
            # Move model to appropriate device
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                print("Model loaded on CUDA device")
            elif torch.backends.mps.is_available():  # For Apple Silicon
                self.model = self.model.to("mps")
                print("Model loaded on MPS device")
            else:
                print("Model loaded on CPU")
                
            print("Model and processor loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe an audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: Transcription text
        """
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            return None
            
        try:
            # Load audio
            audio, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
                
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Process audio
            input_features = self.processor(
                audio, 
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features
            
            # Move to same device as model
            if torch.cuda.is_available():
                input_features = input_features.to("cuda")
            elif torch.backends.mps.is_available():
                input_features = input_features.to("mps")
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription
            
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return None
