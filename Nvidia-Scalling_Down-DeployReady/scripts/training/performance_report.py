import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import numpy as np
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_model(model_path, device, use_flash=False):
    """Load model with optional Flash Attention"""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if use_flash and device.type == 'cuda':
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            )
            print("Model loaded with Flash Attention 2")
        except:
            print("Flash Attention 2 not available, using standard attention")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
    
    model.to(device)
    model.eval()
    return model, tokenizer

def benchmark_inference(model, tokenizer, test_texts, device, num_runs=100, batch_size=1):
    """Benchmark inference performance"""
    print(f"Benchmarking inference: {num_runs} runs, batch_size={batch_size}")
    
    # Warm up
    warmup_text = "This is a warmup text."
    for _ in range(5):
        inputs = tokenizer(warmup_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in tqdm(range(num_runs), desc="Running inference"):
        if batch_size == 1:
            # Single text inference
            text = test_texts[0]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                    _ = model(**inputs)
        else:
            # Batch inference
            batch_texts = test_texts[:batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                    _ = model(**inputs)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_inference = total_time / num_runs
    throughput = num_runs / total_time
    
    return {
        'total_time': total_time,
        'avg_time_per_inference': avg_time_per_inference,
        'throughput': throughput,
        'num_runs': num_runs,
        'batch_size': batch_size
    }

def benchmark_training(model, tokenizer, sample_data, device, num_batches=20):
    """Benchmark training performance"""
    print(f"Benchmarking training: {num_batches} batches")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Warm up
    for _ in range(3):
        batch = sample_data[:8]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.randint(0, 3, (len(batch),)).to(device)
        
        with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in tqdm(range(num_batches), desc="Running training"):
        batch = sample_data[:16]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.randint(0, 3, (len(batch),)).to(device)
        
        with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_batches
    samples_per_second = (num_batches * 16) / total_time
    
    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'samples_per_second': samples_per_second,
        'num_batches': num_batches,
        'batch_size': 16
    }

def test_model_accuracy(model, tokenizer, device):
    """Test model accuracy on sample financial texts"""
    test_cases = [
        ("The company reported strong quarterly earnings growth.", "positive"),
        ("Stock prices plummeted after the earnings report.", "negative"),
        ("The market remained stable throughout the trading session.", "neutral"),
        ("Investors are concerned about the company's debt levels.", "negative"),
        ("The merger announcement boosted shareholder confidence.", "positive"),
        ("Economic indicators suggest a potential recession.", "negative"),
        ("The central bank raised interest rates by 0.5%.", "neutral"),
        ("Corporate profits exceeded analyst expectations.", "positive"),
        ("The housing market shows signs of recovery.", "positive"),
        ("Inflation rates continue to rise above target levels.", "negative")
    ]
    
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    correct = 0
    total = len(test_cases)
    results = []
    
    for text, expected in test_cases:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
        
        predicted = label_mapping[predicted_class]
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        results.append({
            'text': text,
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct
        })
    
    accuracy = correct / total
    return accuracy, results

def generate_performance_report(model_path, output_file="performance_report.json"):
    """Generate comprehensive performance report"""
    set_seed(42)
    device = get_device()
    
    # Test data
    test_texts = [
        "The company reported strong quarterly earnings growth.",
        "Stock prices fell sharply after the announcement.",
        "The market remained stable during trading hours.",
        "Investors are optimistic about future prospects.",
        "Economic indicators show positive trends."
    ]
    
    sample_data = test_texts * 4  # More data for training benchmark
    
    report = {
        'model_path': model_path,
        'device': str(device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmarks': {}
    }
    
    # Test different configurations
    configurations = [
        {'name': 'standard', 'use_flash': False},
        {'name': 'flash_attention', 'use_flash': True}
    ]
    
    for config in configurations:
        print(f"\n{'='*50}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'='*50}")
        
        # Load model
        model, tokenizer = load_model(model_path, device, config['use_flash'])
        
        # Inference benchmarks
        inference_results = {}
        for batch_size in [1, 8, 16]:
            result = benchmark_inference(model, tokenizer, test_texts, device, num_runs=50, batch_size=batch_size)
            inference_results[f'batch_size_{batch_size}'] = result
        
        # Training benchmark
        training_result = benchmark_training(model, tokenizer, sample_data, device, num_batches=10)
        
        # Accuracy test
        accuracy, accuracy_results = test_model_accuracy(model, tokenizer, device)
        
        # Store results
        report['benchmarks'][config['name']] = {
            'inference': inference_results,
            'training': training_result,
            'accuracy': accuracy,
            'accuracy_details': accuracy_results
        }
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nPerformance report saved to: {output_file}")
    return report

def print_summary(report):
    """Print a summary of the performance report"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    for config_name, results in report['benchmarks'].items():
        print(f"\nConfiguration: {config_name.upper()}")
        print("-" * 40)
        
        # Inference summary
        print("Inference Performance:")
        for batch_size, result in results['inference'].items():
            print(f"  {batch_size}: {result['throughput']:.2f} texts/sec "
                  f"({result['avg_time_per_inference']*1000:.2f} ms per text)")
        
        # Training summary
        train_result = results['training']
        print(f"Training Performance: {train_result['samples_per_second']:.2f} samples/sec")
        
        # Accuracy
        print(f"Accuracy: {results['accuracy']*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive performance report")
    parser.add_argument('--model_path', type=str, default='./phrasebank_student_model',
                        help='Path to the trained model')
    parser.add_argument('--output', type=str, default='performance_report.json',
                        help='Output file for the report')
    
    args = parser.parse_args()
    
    # Generate report
    report = generate_performance_report(args.model_path, args.output)
    
    # Print summary
    print_summary(report)

if __name__ == "__main__":
    main() 