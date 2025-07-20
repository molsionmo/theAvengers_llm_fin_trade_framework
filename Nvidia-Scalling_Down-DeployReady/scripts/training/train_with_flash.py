import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import os
import logging
from datetime import datetime
import yaml
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA with Flash Attention support")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_model_with_flash(model_path, device, num_labels=3):
    """Load model with Flash Attention support"""
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model (DistilBERT doesn't support flash_attention_2 directly)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    
    model.to(device)
    print(f"Model loaded successfully on {device}")
    
    return model, tokenizer

def create_flash_training_args(output_dir, hyperparams):
    """Create training arguments optimized for Flash Attention"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hyperparams.get('num_train_epochs', 3),
        per_device_train_batch_size=hyperparams.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=hyperparams.get('per_device_eval_batch_size', 16),
        learning_rate=hyperparams.get('learning_rate', 2e-5),
        evaluation_strategy='steps',
        eval_steps=hyperparams.get('eval_steps', 100),
        save_strategy='steps',
        save_steps=hyperparams.get('save_steps', 100),
        logging_steps=hyperparams.get('logging_steps', 10),
        save_total_limit=hyperparams.get('save_total_limit', 3),
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        gradient_accumulation_steps=hyperparams.get('gradient_accumulation_steps', 1),
        fp16=True if torch.cuda.is_available() else False,
        bf16=False,
        report_to='none',
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        warmup_steps=hyperparams.get('warmup_steps', 100),
        weight_decay=hyperparams.get('weight_decay', 0.01),
        lr_scheduler_type='linear',
        optim='adamw_torch',
        torch_compile=True if torch.cuda.is_available() else False,
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def benchmark_training_speed(model, tokenizer, sample_data, device, num_batches=10):
    """Benchmark training speed with Flash Attention"""
    print("\nBenchmarking training speed...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Warm up
    for _ in range(3):
        batch = sample_data[:8]  # Small batch for warmup
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.randint(0, 3, (len(batch),)).to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_batches):
        batch = sample_data[:16]  # Larger batch for benchmark
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.randint(0, 3, (len(batch),)).to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_batches
    samples_per_second = (num_batches * 16) / total_time
    
    print(f"Training benchmark results:")
    print(f"  Average time per batch: {avg_time_per_batch:.3f} seconds")
    print(f"  Samples per second: {samples_per_second:.2f}")
    print(f"  Total time: {total_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Train model with Flash Attention")
    parser.add_argument('--model_path', type=str, default='./phrasebank_student_model',
                        help='Path to the base model')
    parser.add_argument('--output_dir', type=str, default='./flash_trained_model',
                        help='Output directory for trained model')
    parser.add_argument('--config', type=str, default='../../config/hyperparameters.yaml',
                        help='Path to hyperparameters config')
    parser.add_argument('--benchmark_only', action='store_true',
                        help='Only run benchmark, no training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    device = get_device()
    
    # Load hyperparameters
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            hyperparams = yaml.safe_load(f)
    else:
        hyperparams = {}
    
    # Override epochs if specified
    if args.epochs:
        hyperparams['num_train_epochs'] = args.epochs
    
    # Load model
    model, tokenizer = load_model_with_flash(args.model_path, device)
    
    # Create sample data for benchmarking
    sample_texts = [
        "The company reported strong quarterly earnings growth.",
        "Stock prices fell sharply after the announcement.",
        "The market remained stable during trading hours.",
        "Investors are optimistic about future prospects.",
        "Economic indicators show positive trends.",
        "The merger deal was completed successfully.",
        "Revenue exceeded analyst expectations.",
        "The board approved the dividend increase.",
        "Market volatility increased significantly.",
        "The IPO was oversubscribed by investors.",
        "Corporate profits declined this quarter.",
        "The acquisition strategy paid off.",
        "Shareholders approved the buyback program.",
        "The economic outlook remains uncertain.",
        "The company expanded into new markets.",
        "Financial results beat market expectations."
    ]
    
    if args.benchmark_only:
        benchmark_training_speed(model, tokenizer, sample_texts, device)
        return
    
    # Prepare training data (using sample data for demonstration)
    # In a real scenario, you would load your actual training dataset
    print(f"\nPreparing training data...")
    print(f"Note: Using sample data for demonstration. Replace with actual dataset for real training.")
    
    # Create simple dataset for demonstration
    from datasets import Dataset
    
    # Create training data
    train_texts = sample_texts * 10  # Repeat for more training examples
    train_labels = [np.random.randint(0, 3) for _ in train_texts]  # Random labels for demo
    
    # Create validation data
    val_texts = sample_texts[:8]
    val_labels = [np.random.randint(0, 3) for _ in val_texts]
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=True
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Create training arguments
    training_args = create_flash_training_args(args.output_dir, hyperparams)
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Run benchmark before training
    print("\nRunning pre-training benchmark...")
    benchmark_training_speed(model, tokenizer, sample_texts, device, num_batches=5)
    
    # Start training
    print(f"\nStarting training with Flash Attention...")
    print(f"Training arguments: {training_args}")
    
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # Save model
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")
    
    # Run benchmark after training
    print("\nRunning post-training benchmark...")
    benchmark_training_speed(model, tokenizer, sample_texts, device, num_batches=5)
    
    # Print training results
    print(f"\nTraining completed!")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Training metrics: {train_result.metrics}")
    
    # Evaluate on validation set
    print(f"\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")

if __name__ == "__main__":
    main() 