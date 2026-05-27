import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os
from typing import Dict, List
import bitsandbytes as bnb

class ModelTrainer:
    """
    A class to handle LLM training with LoRA fine-tuning
    """

    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize the model trainer

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.model_name = self.config['model_name']
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        """
        Load the pre-trained model and tokenizer
        """
        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=True,
            device_map="auto",
        )

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA if enabled
        if self.config['use_lora']:
            self._apply_lora()

    def _apply_lora(self):
        """
        Apply LoRA configuration to the model
        """
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, documents: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset for training

        Args:
            documents: List of documents

        Returns:
            Prepared dataset
        """
        # Tokenize documents
        texts = [doc['content'] for doc in documents]

        # Tokenize the texts
        tokenized_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config['max_seq_length'],
            return_tensors="pt"
        )

        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask']
        })

        return dataset

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """
        Train the model

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.config['model_output_dir'],
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=4,
            learning_rate=self.config['learning_rate'],
            num_train_epochs=self.config['num_train_epochs'],
            logging_dir=f"{self.config['model_output_dir']}/logs",
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=True,
            report_to="none"
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        # Start training
        print("Starting training...")
        trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config['model_output_dir'])
        print(f"Model saved to {self.config['model_output_dir']}")

    def save_model(self, path: str = None):
        """
        Save the trained model

        Args:
            path: Path to save the model (uses config path if not provided)
        """
        save_path = path or self.config['model_output_dir']
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()

    # Load model and tokenizer
    trainer.load_model_and_tokenizer()

    # Note: You would need to prepare your dataset before training
    # This is just an example of how to use the trainer
    print("Model trainer initialized successfully!")