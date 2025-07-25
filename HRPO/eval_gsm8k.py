import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm

from utils import *


def evaluate_model(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    use_base_model_only: bool = False,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 1024,
        load_in_4bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if not use_base_model_only and adapter_path:
        model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    dataset = load_dataset('Balaji173/finance_news_sentiment', 'default')['train']
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    elif num_samples is None:
        # 如果没有指定样本数，使用较小的子集进行测试
        dataset = dataset.shuffle(seed=42).select(range(200))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'acc': '0.00%', 'correct': '0'})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data['context'])

        # Prepare prompts using the same format as training
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"What is the sentiment of this tweet? Please choose an answer from {{negative/neutral/positive}}.\nInput: {q.strip()}"},
            ]
            for q in batch_data['context']
        ]

        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        # Generate responses
        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask, 
            generation_config=GenerationConfig(
                do_sample=True,  # for temperature, top-k, etc.
                temperature=temperature,
                max_new_tokens=512,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
        )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(
                tokenizer.special_tokens_map['eos_token']
            )[0]

            # Extract the generated answer for sentiment analysis
            extracted = extract_from_response(response)
            generated_answer = process_finance_sentiment_answer(extracted)
            true_answer = batch_data['target'][j].lower()
            print(generated_answer, true_answer, generated_answer == true_answer)

            # Store the result
            result = {
                'context': batch_data['context'][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': generated_answer == true_answer
            }
            results.append(result)

            if generated_answer == true_answer:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_path': adapter_path,
        'timestamp': datetime.now().isoformat()
    }

    if save_results:
        model_type = "base_model" if use_base_model_only else "trained_model"
        save_path = f"./eval_results_{model_type}_200samples.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_path", type=str, default="./experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.98-temp0.5/checkpoint-2000")
    parser.add_argument("--base_only", action="store_true", help="只评估基础模型，不加载适配器")
    args = parser.parse_args()

    base_model = None
    checkpoint_path = args.checkpoint_path
    
    # 确保checkpoint_path不为None
    if checkpoint_path is None:
        checkpoint_path = "./experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.98-temp0.5/checkpoint-2000"
    
    base_models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    for model in base_models:
        if model.split('/')[-1] in checkpoint_path:
            base_model = model
    
    # 如果没有匹配到模型，使用默认模型
    if base_model is None:
        base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 尝试从路径中提取温度参数，如果失败则使用默认值
    try:
        temperature = float(checkpoint_path.split('-temp')[-1].split('/')[0])
    except (IndexError, ValueError):
        temperature = 0.5
    
    print(checkpoint_path, base_model, temperature)
    print(f"Base model only: {args.base_only}")

    model_type = "base model" if args.base_only else "trained model"
    print(f"Starting finance sentiment evaluation on {model_type}")
    metrics = evaluate_model(
        model_path=base_model,
        adapter_path=checkpoint_path if not args.base_only else None,
        temperature=temperature,
        is_inference=args.greedy,
        batch_size=args.batch_size,
        num_samples=None,
        save_results=True,
        use_base_model_only=args.base_only,
    )