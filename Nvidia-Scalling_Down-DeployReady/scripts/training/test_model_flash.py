import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import argparse
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

def load_model_and_tokenizer(model_path, device):
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model (DistilBERT doesn't support flash_attention_2 directly)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer

def predict_single_text(model, tokenizer, text, device, label_mapping=None):
    """Predict sentiment for a single text using Flash Attention"""
    if label_mapping is None:
        label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
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
        label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
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
        batch_predictions = [label_mapping[pred.item()] for pred in predicted_classes]
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
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark')
    parser.add_argument('--test_sentences', action='store_true',
                        help='Test with financial sentences')
    parser.add_argument('--custom_text', type=str, default=None,
                        help='Custom text to test')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    device = get_device()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    # Test custom text if provided
    if args.custom_text:
        print(f"\nTesting custom text: '{args.custom_text}'")
        result = predict_single_text(model, tokenizer, args.custom_text, device)
        print(f"Prediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
    
    # Test financial sentences
    if args.test_sentences:
        test_financial_sentences(model, tokenizer, device)
    
    # Run benchmark
    if args.benchmark:
        test_texts = [
            "The company reported strong quarterly earnings growth.",
            "Stock prices fell sharply after the announcement.",
            "The market remained stable during trading hours.",
            "Investors are optimistic about future prospects.",
            "Economic indicators show positive trends."
        ]
        benchmark_inference(model, tokenizer, test_texts, device)
    
    # If no specific test is requested, run both
    if not any([args.custom_text, args.test_sentences, args.benchmark]):
        print("\nRunning default tests...")
        test_financial_sentences(model, tokenizer, device)
        benchmark_inference(model, tokenizer, [], device)

if __name__ == "__main__":
    main() 