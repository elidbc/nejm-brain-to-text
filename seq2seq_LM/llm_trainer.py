import json
import torch
import sys
import numpy as np
import evaluate
import re
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,    
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

MODEL_ID = 'google/byt5-base'
TRAIN_DATA_FILE = '../data/llm_training_data/train.jsonl'
TRAIN_SYNTH_DATA_FILE = '../data/llm_training_data/train_synth.jsonl'
VAL_DATA_FILE = '../data/llm_training_data/val.jsonl'
VAL_SYNTH_DATA_FILE = '../data/llm_training_data/val_synth.jsonl'
OUTPUT_DIR = 'trained_models/v3'
NOISE_PROFILE_FILE = '../data/llm_training_data/GRU_confusion_matrix.json'

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

def train():
    val_data = []

    print(f"Loading data from {TRAIN_DATA_FILE} and {VAL_DATA_FILE}")
    # Load clean data, and clean + synth val data
    with open(TRAIN_DATA_FILE, 'r') as f:
        train_clean = [json.loads(line) for line in f]
    #with open(VAL_DATA_FILE, 'r') as f:
        #val_data += [json.loads(line) for line in f]
    with open(VAL_SYNTH_DATA_FILE, 'r') as f:
        val_data += [json.loads(line) for line in f]
    
    # Get Noise Profile
    with open(NOISE_PROFILE_FILE, 'r') as f:
        noise_profile = json.load(f)

    def data_noiser(phoneme_str, profile):
        tokens = phoneme_str.split()
        new_tokens = []
        insert_dist = profile.get("<INSERTIONS>", {}).get("distribution", [])
        if insert_dist:
            ins_candidates = [x['phoneme'] for x in insert_dist]
            ins_weights = [x['prob'] for x in insert_dist]

        for t in tokens:
            # --- A. INSERTION NOISE (Hallucinations) ---
            # Randomly insert a phoneme *before* the current one
            # Rate: 1% (adjust based on how hallucinatory your GRU is)
            if random.random() < 0.005: 
                new_tokens.append(random.choices(ins_candidates, weights=ins_weights, k=1)[0])

            # --- B. SEP PROTECTION ---
            if t == "SEP":
                # SEP is structural. Only mess with it rarely (e.g. 1% chance)
                # unless your profile explicitly shows high SEP error rates.
                if random.random() > 0.99:
                     pass # Delete SEP
                else:
                    new_tokens.append(t)
                continue

            # --- C. SUBSTITUTION / DELETION ---
            if t in profile and t != "<INSERTIONS>":
                stats = profile[t]
                
                # The confusion matrix gives us the exact probability this phoneme is CORRECT
                # e.g., "Y" might be correct 89.9% of the time.
                correct_prob = stats.get('correct_prob', 0.9)
                
                # Roll to see if we keep it clean
                if random.random() < correct_prob:
                    new_tokens.append(t)
                else:
                    # We are in the error zone. Choose a specific error.
                    mappings = stats.get('mappings', [])
                    if not mappings:
                        # Fallback if mappings list is empty but error rate > 0
                        new_tokens.append(t) 
                        continue
                        
                    err_candidates = [m['error'] for m in mappings]
                    err_weights = [m['prob'] for m in mappings]
                    
                    # Normalize weights to sum to 1
                    total_w = sum(err_weights)
                    if total_w > 0:
                        norm_weights = [w/total_w for w in err_weights]
                        choice = random.choices(err_candidates, weights=norm_weights, k=1)[0]
                        
                        if choice == "<DELETE>":
                            continue
                        else:
                            new_tokens.append(choice)
                    else:
                        new_tokens.append(t)
            else:
                # Token not in profile (rare phoneme). 
                # Apply a small generic noise (e.g. 5%) or keep clean.
                if random.random() < 0.95:
                    new_tokens.append(t)
                else:
                    # Randomly drop rare phonemes occasionally
                    if random.random() < 0.5:
                        continue 
                    new_tokens.append(t)

        return " ".join(new_tokens)

    print("Generating synthetic data...")
    train_augmented = []
    for _ in range(8):
        for example in train_clean:
            noisy_input = data_noiser(example['input'], noise_profile)
            train_augmented.append({
                "input": noisy_input,
                "target": example['target'],
            })
    print(f"Generated {len(train_augmented)} synthetic examples")
    train_data = train_augmented

    # Save the full training dataset to v2 folder
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_data_path = os.path.join(OUTPUT_DIR, "QLoRA_v3_train_data.json")
    with open(train_data_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved {len(train_data)} training examples to {train_data_path}")

    random.shuffle(train_data)
    random.shuffle(val_data)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )

    print(f"Laoding model in 4-bit")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r = 64,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['target']

        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True) #, padding="max_length")
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH , truncation=True) # , padding="max_length")

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
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

        def normalize_text(text):
            text = text.lower()
            text = re.sub(r"[^\w\s']", '', text)
            return text
        
        # --- ROUGE ---
        norm_preds = [normalize_text(pred) for pred in decoded_preds]
        norm_labels = [normalize_text(label) for label in decoded_labels]
        result = metric_rouge.compute(predictions=norm_preds, references=norm_labels, use_stemmer=True)
        
        # --- WER ---
        # WER is sensitive to casing and punctuation. 
        # Usually, we verify "standardized" WER (lowercase, no punctuation) 
        # but for LLM generation, raw WER is often fine. 
        # If errors happen due to empty strings, we handle them safely.
        try:
            wer_score = metric_wer.compute(predictions=norm_preds, references=norm_labels)
        except ValueError as e:
            # Fallback if predictions are empty
            print(f"Error calculating WER: {e}")
            wer_score = 1.0 

        # Combine results
        final_metrics = {k: round(v * 100, 4) for k, v in result.items()} # ROUGE is 0-100 scale usually
        final_metrics["wer"] = round(wer_score * 100, 4) # Convert WER to percentage for consistency
        
        return final_metrics
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model, 
        padding=True,
        label_pad_token_id=-100
    )

    # optimize for T4 GPU
    training_args = Seq2SeqTrainingArguments(
        # optimization
        output_dir=OUTPUT_DIR,
        optim="paged_adamw_32bit",
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.05,

        # Speed
        fp16=False, 
        gradient_checkpointing=False,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,

        # Stability
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=250,
        save_steps=250,
        per_device_eval_batch_size=64,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="wandb",
        run_name="v3-8x-synth"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete — Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved to", OUTPUT_DIR)

def eval_model(checkpoint_path: str = "trained_models/checkpoint-2525", num_examples: int = 10):
    """
    Evaluate the fine-tuned model on the validation set.
    Computes ROUGE and WER metrics, and prints randomly selected examples.
    """
    from peft import PeftModel
    
    # Load validation data
    val_data = []
    print(f"Loading validation data from {VAL_DATA_FILE}")
    #with open(VAL_DATA_FILE, 'r') as f:
        #val_data += [json.loads(line) for line in f]
    with open(VAL_SYNTH_DATA_FILE, 'r') as f:
        val_data += [json.loads(line) for line in f]
    
    print(f"Total validation samples: {len(val_data)}")
    
    # Load tokenizer from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load base model
    print(f"Loading base model: {MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load PEFT adapter
    print(f"Loading PEFT adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    # Initialize metrics
    metric_rouge = evaluate.load("rouge")
    metric_wer = evaluate.load("wer")
    
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r"[^\w\s']", '', text)
        return text
    
    # Run predictions on all validation data
    print("\nRunning predictions on validation set...")
    all_predictions = []
    all_references = []
    
    batch_size = 16
    for i in range(0, len(val_data), batch_size):
        batch = val_data[i:i+batch_size]
        inputs = [item['input'] for item in batch]
        targets = [item['target'] for item in batch]
        
        # Tokenize inputs
        input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).input_ids.to(model.device)
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        all_predictions.extend(decoded_preds)
        all_references.extend(targets)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(val_data))}/{len(val_data)} samples...")
    
    # Normalize for metrics
    norm_preds = [normalize_text(pred) for pred in all_predictions]
    norm_refs = [normalize_text(ref) for ref in all_references]
    
    # Compute ROUGE
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    rouge_result = metric_rouge.compute(predictions=norm_preds, references=norm_refs, use_stemmer=True)
    for key, value in rouge_result.items():
        print(f"  {key}: {value * 100:.2f}%")
    
    # Compute WER
    try:
        wer_score = metric_wer.compute(predictions=norm_preds, references=norm_refs)
        print(f"  WER: {wer_score * 100:.2f}%")
    except ValueError as e:
        print(f"  WER: Error - {e}")
    
    # Print random examples
    print("\n" + "="*60)
    print(f"RANDOM EXAMPLES ({num_examples} samples)")
    print("="*60)
    
    indices = random.sample(range(len(val_data)), min(num_examples, len(val_data)))
    
    for idx in indices:
        phonemes = val_data[idx]['input']
        ground_truth = val_data[idx]['target']
        prediction = all_predictions[idx]
        
        print(f"\n--- Example {indices.index(idx) + 1} ---")
        print(f"PHONEMES:     {phonemes}")
        print(f"GROUND TRUTH: {ground_truth}")
        print(f"PREDICTION:   {prediction}")
        
        # Show match status
        is_exact_match = normalize_text(prediction) == normalize_text(ground_truth)
        print(f"EXACT MATCH:  {'✓ YES' if is_exact_match else '✗ NO'}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


def longest_seq():
    with open(TRAIN_DATA_FILE, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(VAL_DATA_FILE, 'r') as f:
        val_data = [json.loads(line) for line in f]
    with open(TRAIN_SYNTH_DATA_FILE, 'r') as f:
        train_synth_data = [json.loads(line) for line in f]
    with open(VAL_SYNTH_DATA_FILE, 'r') as f:
        val_synth_data = [json.loads(line) for line in f]

    all_data = train_data + val_data + train_synth_data + val_synth_data
    max_input_len = max(len(x['input'].encode('utf-8')) for x in all_data)
    max_target_len = max(len(x['target'].encode('utf-8')) for x in all_data)

    print(f"Longest input sequence length: {max_input_len}")
    print(f"Longest target sequence length: {max_target_len}")

if __name__ == "__main__":
    
    #checkpoint = "trained_models/v2/checkpoint-2525"
    #eval_model(checkpoint_path=checkpoint)

    train()


    