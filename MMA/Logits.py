"""
pip install transformers>=4.40.0 torch>=2.1.0
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True
        )

prompt = "请用一句话介绍量子计算。"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
    outputs = model(input_ids=input_ids)

# 取第 20 层 hidden state
hidden_20 = outputs.hidden_states[20]
print(">>> Hidden State (layer 20):")
print("shape :", hidden_20.shape)
print("mean  :", hidden_20.float().mean().item())
print("std   :", hidden_20.float().std().item())

# logits
logits = outputs.logits
print("\n>>> Logits:")
print("shape :", logits.shape)
print("max   :", logits.max().item())
print("min   :", logits.min().item())

# 生成回复
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=1024,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)
print("\n>>> 模型回复：\n", response)