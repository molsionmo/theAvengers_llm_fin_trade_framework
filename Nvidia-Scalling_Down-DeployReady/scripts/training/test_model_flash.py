#!/usr/bin/env python3
"""
Test script for trained sentiment analysis models.

Usage:
    # Evaluate on test dataset
    python test_model_flash.py --evaluate
    
    # Test custom text
    python test_model_flash.py --custom_text "The company reported excellent earnings."
    
    # Run benchmark
    python test_model_flash.py --benchmark
    
    # Test with financial sentences
    python test_model_flash.py --test_sentences
    
    # Run all tests
    python test_model_flash.py --all
    
    # Specify different model path
    python test_model_flash.py --model_path ./my_model --evaluate
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import argparse
from tqdm import tqdm
import pickle
import os

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

def get_label_mapping(model):
    """Get label mapping from model configuration"""
    if hasattr(model.config, 'id2label') and model.config.id2label:
        return model.config.id2label
    elif hasattr(model.config, 'num_labels'):
        # Default mapping for sentiment analysis
        if model.config.num_labels == 3:
            return {0: 'negative', 1: 'neutral', 2: 'positive'}
        elif model.config.num_labels == 2:
            return {0: 'negative', 1: 'positive'}
        else:
            return {i: f'label_{i}' for i in range(model.config.num_labels)}
    else:
        # Fallback
        return {0: 'negative', 1: 'neutral', 2: 'positive'}

def load_model_and_tokenizer(model_path, device):
    print(f"Loading model from: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded successfully")
        
        # Load model with proper configuration
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32
        )
        
        model.to(device)
        model.eval()
        
        # Print model info
        print(f"✅ Model loaded successfully on {device}")
        print(f"Model type: {model.__class__.__name__}")
        print(f"Number of labels: {model.config.num_labels}")
        
        # Check if model has label mappings
        if hasattr(model.config, 'id2label') and model.config.id2label:
            print(f"Label mappings: {model.config.id2label}")
            
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def predict_single_text(model, tokenizer, text, device, label_mapping=None):
    """Predict sentiment for a single text using Flash Attention"""
    if label_mapping is None:
        label_mapping = get_label_mapping(model)
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Enable Flash Attention for inference
    with torch.no_grad():
        with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            predicted_class = int(predicted_class)  # Ensure it's an integer
            confidence = probabilities[0][predicted_class].item()
    
    return {
        'predicted_label': label_mapping[predicted_class],
        'confidence': confidence,
        'probabilities': {
            label_mapping[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
    }

def batch_predict(model, tokenizer, texts, device, batch_size=8, label_mapping=None):
    """Batch prediction with Flash Attention for better efficiency"""
    if label_mapping is None:
        label_mapping = get_label_mapping(model)
    
    predictions = []
    confidences = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict with Flash Attention
        with torch.no_grad():
            with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(logits, dim=-1)
                batch_confidences = torch.max(probabilities, dim=-1)[0]
        
        # Convert to labels
        batch_predictions = [label_mapping[int(pred.item())] for pred in predicted_classes]
        batch_confidences = batch_confidences.cpu().numpy()
        
        predictions.extend(batch_predictions)
        confidences.extend(batch_confidences)
    
    return predictions, confidences

def benchmark_inference(model, tokenizer, test_texts, device, num_runs=100):
    """Benchmark inference speed with Flash Attention"""
    print(f"\nBenchmarking inference speed with {num_runs} runs...")
    
    # Warm up
    warmup_text = "This is a warmup text for benchmarking."
    for _ in range(10):
        predict_single_text(model, tokenizer, warmup_text, device)
    
    # Benchmark single text inference
    test_text = "The company reported strong quarterly earnings growth."
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_runs):
        predict_single_text(model, tokenizer, test_text, device)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = 1.0 / avg_time
    
    print(f"Single text inference:")
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} texts/second")
    
    # Benchmark batch inference
    batch_size = 16
    batch_texts = [test_text] * batch_size
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_runs // 4):  # Fewer runs for batch
        batch_predict(model, tokenizer, batch_texts, device, batch_size)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_texts = (num_runs // 4) * batch_size
    avg_batch_time = (end_time - start_time) / total_texts
    batch_throughput = 1.0 / avg_batch_time
    
    print(f"Batch inference (batch_size={batch_size}):")
    print(f"  Average time per text: {avg_batch_time*1000:.2f} ms")
    print(f"  Throughput: {batch_throughput:.2f} texts/second")
    print(f"  Speedup: {batch_throughput/throughput:.2f}x")

def load_test_data(dataset_path='./test_bert_phrasebank.pkl'):
    """Load test data for evaluation"""
    try:
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                test_data = pickle.load(f)
            print(f"✅ Loaded test data with {len(test_data)} samples")
            return test_data
        else:
            print(f"❌ Test data file not found: {dataset_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

def evaluate_on_test_data(model, tokenizer, device, test_data_path='./test_bert_phrasebank.pkl'):
    """Evaluate model on test dataset"""
    test_data = load_test_data(test_data_path)
    if test_data is None:
        return
    
    print("\n" + "="*60)
    print("Evaluating on Test Dataset")
    print("="*60)
    
    # Handle Hugging Face Dataset format
    if hasattr(test_data, 'to_pandas'):
        # Convert HF Dataset to pandas for easier handling
        df = test_data.to_pandas()
        # Decode the input_ids back to text for human-readable display
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in df['input_ids']]
        true_labels = df['labels'].tolist()
    else:
        # Original format
        texts = [item['text'] for item in test_data]
        true_labels = [item['label'] for item in test_data]
    
    # Get label mapping
    label_mapping = get_label_mapping(model)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    print(f"Dataset size: {len(texts)} samples")
    print(f"Label mapping: {label_mapping}")
    
    # Use the pre-tokenized data for prediction if available
    if hasattr(test_data, 'to_pandas'):
        # Direct evaluation using pre-tokenized data
        pred_numeric, confidences = evaluate_pretokenized_data(model, test_data, device, batch_size=32)
        # Convert to string labels for display
        predictions = [label_mapping[pred] for pred in pred_numeric]
    else:
        # Batch prediction for raw text
        predictions, confidences = batch_predict(model, tokenizer, texts, device, batch_size=32)
        # Convert predictions to numeric labels for evaluation
        pred_numeric = [reverse_mapping[pred] for pred in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_numeric)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_numeric, average='weighted', zero_division=0
    )
    
    print(f"\n📊 Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Detailed classification report
    target_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    print(f"\n📈 Detailed Classification Report:")
    print(classification_report(true_labels, pred_numeric, target_names=target_names, zero_division=0))
    
    # Show some examples
    print(f"\n📝 Sample Predictions:")
    indices = np.random.choice(len(texts), min(10, len(texts)), replace=False)
    for i, idx in enumerate(indices):
        true_label_name = label_mapping[true_labels[idx]]
        pred_label_name = predictions[idx]
        confidence = confidences[idx]
        correct = "✅" if true_label_name == pred_label_name else "❌"
        
        print(f"{i+1:2d}. {correct} Text: {texts[idx][:80]}...")
        print(f"     True: {true_label_name} | Pred: {pred_label_name} (conf: {confidence:.3f})")
        print()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'confidences': confidences
    }

def evaluate_pretokenized_data(model, dataset, device, batch_size=32):
    """Evaluate model on pre-tokenized Hugging Face dataset"""
    predictions = []
    confidences = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        # Get batch data
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        # Prepare inputs
        inputs = {
            'input_ids': torch.tensor(batch['input_ids']).to(device),
            'attention_mask': torch.tensor(batch['attention_mask']).to(device)
        }
        
        # Predict
        with torch.no_grad():
            with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(logits, dim=-1)
                batch_confidences = torch.max(probabilities, dim=-1)[0]
        
        # Convert to CPU and extend lists
        predictions.extend(predicted_classes.cpu().numpy())
        confidences.extend(batch_confidences.cpu().numpy())
    
    return predictions, confidences

def test_financial_sentences(model, tokenizer, device):
    """Test the model with various financial sentences"""
    test_sentences = [
        "The company's revenue increased by 25% this quarter.",
        "Stock prices plummeted after the earnings report.",
        "The market remained stable throughout the trading session.",
        "Investors are concerned about the company's debt levels.",
        "The merger announcement boosted shareholder confidence.",
        "Economic indicators suggest a potential recession.",
        "The central bank raised interest rates by 0.5%.",
        "Corporate profits exceeded analyst expectations.",
        "The housing market shows signs of recovery.",
        "Inflation rates continue to rise above target levels."
    ]
    
    print("\n" + "="*60)
    print("Testing Financial Sentiment Analysis")
    print("="*60)
    
    results = []
    for i, sentence in enumerate(test_sentences, 1):
        result = predict_single_text(model, tokenizer, sentence, device)
        results.append({
            'sentence': sentence,
            'prediction': result['predicted_label'],
            'confidence': result['confidence']
        })
        
        print(f"{i:2d}. {sentence}")
        print(f"    Prediction: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
        print()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test trained model with Flash Attention")
    parser.add_argument('--model_path', type=str, default='./phrasebank_student_model',
                        help='Path to the trained model')
    parser.add_argument('--test_data', type=str, default='./test_bert_phrasebank.pkl',
                        help='Path to test dataset (pickle file)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark')
    parser.add_argument('--test_sentences', action='store_true',
                        help='Test with financial sentences')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate on test dataset')
    parser.add_argument('--custom_text', type=str, default=None,
                        help='Custom text to test')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    device = get_device()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    # Print model information
    label_mapping = get_label_mapping(model)
    print(f"\n🏷️  Model Labels: {label_mapping}")
    
    # Test custom text if provided
    if args.custom_text:
        print(f"\n🔍 Testing custom text: '{args.custom_text}'")
        result = predict_single_text(model, tokenizer, args.custom_text, device)
        print(f"Prediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
    
    # Evaluate on test dataset
    if args.evaluate or args.all:
        evaluate_on_test_data(model, tokenizer, device, args.test_data)
    
    # Test financial sentences
    if args.test_sentences or args.all:
        test_financial_sentences(model, tokenizer, device)
    
    # Run benchmark
    if args.benchmark or args.all:
        test_texts = [
            "The company reported strong quarterly earnings growth.",
            "Stock prices fell sharply after the announcement.",
            "The market remained stable during trading hours.",
            "Investors are optimistic about future prospects.",
            "Economic indicators show positive trends."
        ]
        benchmark_inference(model, tokenizer, test_texts, device)
    
    # If no specific test is requested, run evaluation by default
    if not any([args.custom_text, args.test_sentences, args.benchmark, args.evaluate, args.all]):
        print("\n🚀 Running default evaluation...")
        evaluate_on_test_data(model, tokenizer, device, args.test_data)

if __name__ == "__main__":
    main() 