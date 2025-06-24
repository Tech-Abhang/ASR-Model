#testing my test branch

from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

def load_asr_model():
    # Load the fine-tuned model directly
    try:
        # Load model and processor separately for better control
        processor = WhisperProcessor.from_pretrained("./whisper-finetuned")
        model = WhisperForConditionalGeneration.from_pretrained("./whisper-finetuned")
        
        # Clear problematic generation config
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = None
        
        # Create pipeline with the modified model
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=False
        )
        return pipe
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        # Fallback to base whisper model
        model = pipeline("automatic-speech-recognition", model="openai/whisper-small")
        return model