import json
import os
import random
import time
from collections import defaultdict
from tqdm import tqdm
import requests
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# configuration
# adjust this path to your directory
base_path = "./politicalcompassdata/"

political_coordinates_file = os.path.join(base_path, "political_ideologies_coordinates.jsonl")
trained_model_dir = os.path.join(base_path, "distilbert_checkpoints")

# transformer model for fine-tuning
transformer_model_name = 'distilbert-base-uncased'
tokenizer = None
# other options: 
# 'distilbert-base-uncased' (fast, memory-efficient)
# 'roberta-base' / 'roberta-large' (larger, slower, more vram; may need per_device_train_batch_size=2, gradient_accumulation_steps=2, gradient_checkpointing=true)
# 'bert-base-uncased' (similar to roberta in size/speed considerations)

# custom pytorch dataset
class politicalcompassdataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # tokenize single text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def run_predictor_and_interpreter():
    # load ideology data
    print("loading ideology data with coordinates from phase 3...")

    original_ideologies_data = []
    try:
        with open(political_coordinates_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="loading original coordinates data"):
                data = json.loads(line.strip())
                if data.get("coordinates") and data["coordinates"]["x"] is not None and data["coordinates"]["y"] is not None:
                    original_ideologies_data.append(data)
                else:
                    print(f"skipping '{data.get('ideology_name', 'unknown')}' due to missing or invalid coordinates.")
        
        if not original_ideologies_data:
            print(f"error: no ideologies with valid coordinates found. ensure phase 3 completed successfully.")
            return

    except filenotfounderror:
        print(f"error: '{political_coordinates_file}' not found. ensure phase 3 completed successfully.")
        return
    except json.jsondecodeerror as e:
        print(f"error: could not decode json from '{political_coordinates_file}': {e}")
        return
    except Exception as e:
        print(f"unexpected error loading data: {e}")
        return

    print(f"loaded {len(original_ideologies_data)} original ideologies for training.")

    # data preparation
    descriptions = [ideology_data["article_body"] for ideology_data in original_ideologies_data]
    coordinates = np.array([[ideology_data["coordinates"]["x"], ideology_data["coordinates"]["y"]] for ideology_data in original_ideologies_data])
    
    print(f"training on original dataset size: {len(descriptions)} samples.")

    # load transformer tokenizer
    print("\nloading transformer tokenizer...")
    try:
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        print(f"tokenizer loaded. tokenization will happen on-the-fly (max_length=512).")
    except Exception as e:
        print(f"error loading tokenizer: {e}")
        print("ensure 'transformers' library is installed and internet connection is available.")
        return

    # create pytorch dataset
    train_dataset = politicalcompassdataset(
        texts=descriptions, 
        labels=coordinates, 
        tokenizer=tokenizer, 
        max_length=512
    )
    print(f"created training dataset with {len(train_dataset)} samples.")

    # load transformer model for regression
    print("\nloading transformer model for regression...")
    try:
        model_for_training = AutoModelForSequenceClassification.from_pretrained(
            transformer_model_name, 
            num_labels=2, 
            problem_type="regression"
        )
        print(f"model loaded: {transformer_model_name} with {model_for_training.num_parameters()} parameters.")
    except Exception as e:
        print(f"error loading model: {e}")
        print("ensure 'transformers' library is installed and internet connection is available.")
        return

    # define training arguments
    training_args = TrainingArguments(
        output_dir=trained_model_dir,
        num_train_epochs=20,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(trained_model_dir, "logs"),
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        fp16=True,
        learning_rate=1e-5,
        report_to="none",
        gradient_checkpointing=False,
        load_best_model_at_end=False,
    )

    # initialize trainer
    trainer = Trainer(
        model=model_for_training,
        args=training_args,
        train_dataset=train_dataset,
    )

    # check for existing checkpoints
    last_checkpoint = None
    if os.path.isdir(trained_model_dir):
        checkpoints = [d for d in os.listdir(trained_model_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            last_checkpoint = max(os.path.join(trained_model_dir, d) for d in checkpoints)
            print(f"\nresuming training from checkpoint: {last_checkpoint}")
        else:
            print("\nno existing checkpoints found. starting training from scratch.")
    else:
        print("\nno existing model directory found. starting training from scratch.")

    # train model
    print("\nstarting model fine-tuning (feasible rigorous training on original data)...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("model fine-tuning complete.")

    # save final fine-tuned model
    try:
        final_model_save_path = os.path.join(trained_model_dir, "final_model")
        os.makedirs(final_model_save_path, exist_ok=True)
        model_for_training.save_pretrained(final_model_save_path)
        tokenizer.save_pretrained(final_model_save_path)
        print(f"final fine-tuned model and tokenizer saved to '{final_model_save_path}'.")
    except Exception as e:
        print(f"error saving final fine-tuned model: {e}")

    print("\n--- phase 4 complete ---")
    print(f"fine-tuned model and tokenizer are saved to '{os.path.join(trained_model_dir, 'final_model')}'.")
    print(f"original coordinate data from phase 3 is in '{political_coordinates_file}'.")
    print("these files are now ready for the local predictor with knn.")

if __name__ == "__main__":
    print("starting program 7: political compass predictor (pycharm version)")
    run_predictor_and_interpreter()
