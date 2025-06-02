"""
Local fine-tuning client using Hugging Face transformers Seq2SeqTrainer.
"""
import os
import json
import torch
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
import dataclasses
from torch.utils.data import Dataset

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any

    def __call__(self, features):
        # Optimize batch processing for speed
        batch_size = len(features)
        
        # Extract input features and labels separately for efficient processing
        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # Batch process audio inputs
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f} for f in input_features], 
            return_tensors="pt"
        )
        
        # Batch process labels
        label_features = [{"input_ids": l} for l in labels]
        labels_batch = self.processor.tokenizer.pad(
            label_features, 
            return_tensors="pt",
            padding=True
        )
        
        # Efficiently mask padding tokens
        labels_tensor = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]
        labels_tensor = labels_tensor.masked_fill(attention_mask.ne(1), -100)
        
        # Handle BOS token removal if present
        if batch_size > 0 and labels_tensor.size(1) > 0:
            if (labels_tensor[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels_tensor = labels_tensor[:, 1:]
        
        batch["labels"] = labels_tensor
        return batch

class WhisperDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_features": sample["input_features"],
            "labels": sample["labels"]
        }

class FineTuningClient:
    def __init__(self, model_path="./whisper-small"):
        """
        Initialize the fine-tuning client.
        
        Args:
            model_path (str): Path to the local model directory
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        
    def load_model_and_processor(self):
        """Load the model and processor with optimizations."""
        print(f"Loading model from {self.model_path}...")
        
        # Load model
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # Move model to MPS if available
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            self.model = self.model.to(device)
            print(f"Model moved to MPS device: {device}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            self.model = self.model.to(device)
            print(f"Model moved to CUDA device: {device}")
        else:
            print("Using CPU device")
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        print("Model and processor loaded successfully.")
    
    def prepare_training_arguments(self, output_dir="./whisper-finetuned", 
                                 per_device_train_batch_size=8, 
                                 gradient_accumulation_steps=1,
                                 learning_rate=2e-5, 
                                 num_train_epochs=3,
                                 warmup_steps=100,
                                 has_eval_dataset=False):
        """
        Prepare Seq2SeqTrainingArguments with optimizations for speed.
        
        Args:
            output_dir (str): Directory to save the fine-tuned model
            per_device_train_batch_size (int): Batch size per device
            gradient_accumulation_steps (int): Gradient accumulation steps
            learning_rate (float): Learning rate
            num_train_epochs (int): Number of training epochs
            warmup_steps (int): Number of warmup steps
            has_eval_dataset (bool): Whether evaluation dataset is available
            
        Returns:
            Seq2SeqTrainingArguments: Training arguments
        """
        # Adjust evaluation strategy based on whether we have eval data
        if has_eval_dataset:
            eval_strategy = "steps"
            eval_steps = 50  # More frequent eval for faster feedback
        else:
            eval_strategy = "no"
            eval_steps = None
        
        # Optimize for Apple Silicon MPS
        use_mps = torch.backends.mps.is_available()
        use_fp16 = torch.cuda.is_available()  # Keep fp16 only for CUDA
        
        # Enable gradient checkpointing for memory efficiency
        gradient_checkpointing = True
        
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            fp16=use_fp16,
            eval_strategy=eval_strategy,
            save_steps=50,  # Save more frequently
            eval_steps=eval_steps,
            logging_steps=10,  # More frequent logging
            save_total_limit=2,
            predict_with_generate=True,
            generation_max_length=225,
            report_to="none",
            remove_unused_columns=False,
            # Speed optimizations
            gradient_checkpointing=gradient_checkpointing,
            dataloader_num_workers=4,  # Parallel data loading
            dataloader_pin_memory=True,
            group_by_length=False,  # Disable to avoid input_ids key requirement
            # Reduce evaluation overhead
            per_device_eval_batch_size=per_device_train_batch_size * 2,
            eval_accumulation_steps=1,
            # Learning rate scheduler optimization
            lr_scheduler_type="cosine",
            weight_decay=0.01,
        )
    
    def create_trainer(self, training_args, train_dataset, eval_dataset=None):
        """
        Create Seq2SeqTrainer.
        
        Args:
            training_args: Seq2SeqTrainingArguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Seq2SeqTrainer: Configured trainer
        """
        if not self.model or not self.processor:
            raise ValueError("Model and processor must be loaded first. Call load_model_and_processor().")
        
        # Create data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        return Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
        )
    
    def fine_tune(self, preprocessed_samples, validation_samples=None, **training_kwargs):
        """
        Fine-tune the model using preprocessed samples.
        
        Args:
            preprocessed_samples (list): List of preprocessed training samples
            validation_samples (list): List of preprocessed validation samples
            **training_kwargs: Additional training arguments
            
        Returns:
            str: Path to the fine-tuned model
        """
        # Load model and processor
        self.load_model_and_processor()
        
        # Create datasets
        train_dataset = WhisperDataset(preprocessed_samples)
        eval_dataset = WhisperDataset(validation_samples) if validation_samples else None
        
        # Prepare training arguments
        training_args = self.prepare_training_arguments(
            has_eval_dataset=eval_dataset is not None,
            **training_kwargs
        )
        
        # Create trainer
        trainer = self.create_trainer(training_args, train_dataset, eval_dataset)
        
        # Start training
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        print(f"Fine-tuning completed. Model saved to {training_args.output_dir}")
        
        return training_args.output_dir
