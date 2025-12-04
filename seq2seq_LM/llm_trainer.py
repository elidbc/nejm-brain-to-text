import json
import torch
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,    
)

MODEL_ID = 'google/byt5-base'
TRAIN_DATA_FILE = 'data/llm_training_data/train.jsonl'
VAL_DATA_FILE = 'data/llm_training_data/val.jsonl'

def train():
    with open(TRAIN_DATA_FILE, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(VAL_DATA_FILE, 'r') as f:
        val_data = [json.loads(line) for line in f]

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    def preprocess_function(examples):
        inputs = exampels['input']
        targets = examples['target']

        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        labels = tokenizer(targets, max_length=512, truncation=True)

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100 for l in label) for label in labels["input_ids"]]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Tokenizing data...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

    print("Training data size:", len(tokenized_train_dataset))
    print("Validation data size:", len(tokenized_val_dataset))

    metric_rouge = evaluate.load("rouge")
    metric_wer = evaluate.load("wer")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Decode generated tokens
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Decode labels (replacing -100 with pad token to avoid errors)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # --- ROUGE ---
        result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Convert to standard dictionary format
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        
        # --- WER ---
        # WER is sensitive to casing and punctuation. 
        # Usually, we verify "standardized" WER (lowercase, no punctuation) 
        # but for LLM generation, raw WER is often fine. 
        # If errors happen due to empty strings, we handle them safely.
        try:
            wer_score = metric_wer.compute(predictions=decoded_preds, references=decoded_labels)
        except ValueError:
            # Fallback if predictions are empty
            wer_score = 1.0 

        # Combine results
        final_metrics = {k: round(v * 100, 4) for k, v in result.items()} # ROUGE is 0-100 scale usually
        final_metrics["wer"] = round(wer_score * 100, 4) # Convert WER to percentage for consistency
        
        return final_metrics
    
    # optimize for T4 GPU
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=4e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete — Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    train()


    