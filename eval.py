import argparse
import json
import os
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
from prompts import BASE_SYSTEM_PROMPT, QA_TEMPLATE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged HF model")
    parser.add_argument("--dataset_name", type=str, default="cheryyunl/MomaGraph-Bench", help="Dataset path")
    parser.add_argument("--output_file", type=str, default="eval_results.jsonl")
    parser.add_argument("--batch_size", type=int, default=10000) # Process all at once if memory allows, or define chunk size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()

def format_prompt(item):
    # Construct the content part
    # Note: item['options'] is likely a string or list. We need to format it nicely.
    options_str = item['options']
    if isinstance(options_str, list):
        options_str = ", ".join(options_str)
    
    user_content = QA_TEMPLATE.format(
        task_instruction=item['task_instruction'],
        question=item['question'],
        options=options_str
    )
    
    # Combine with system prompt
    # Using Qwen chat template format generally
    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"}, # Image data is passed via multi_modal_data
            {"type": "text", "text": user_content}
        ]}
    ]
    return messages

def extract_choice(response_text):
    # Look for "Final Choice: (X)"
    match = re.search(r"Final Choice:\s*\(?([A-D])\)?", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: look for just (A), (B), etc. at the end
    matches = re.findall(r"\(?([A-D])\)?", response_text)
    if matches:
        return matches[-1].upper()
        
    return None

import base64
import io

def image_to_data_url(image):
    # Resize if too large to avoid OOM and slow inference
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
        
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    llm = LLM(
        model=args.model_path, 
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=16384,  # Balanced setting
        limit_mm_per_prompt={"image": 4} 
    )
    
    # Processor not needed for chat API
    
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train") 
    
    print(f"Processing {len(dataset)} samples...")
    
    # Prepare inputs
    inputs = []
    ground_truths = []
    
    for item in dataset:
        # Convert PIL image to base64 data URL for vLLM compatibility
        image_url = image_to_data_url(item['image'])
        
        messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}}, 
                {"type": "text", "text": QA_TEMPLATE.format(
                    task_instruction=item['task_instruction'],
                    question=item['question'],
                    options=", ".join(item['options']) if isinstance(item['options'], list) else item['options']
                )}
            ]}
        ]
        
        inputs.append(messages)
        ground_truths.append(item['correct_answer'])

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=["<|endoftext|>", "<|im_end|>"]
    )
    
    print("Generating responses...")
    
    # Use chat API with image_url
    outputs = llm.chat(
        messages=inputs,
        sampling_params=sampling_params
    )

    
    correct_count = 0
    results = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gt = ground_truths[i]
        
        # 1. Extract Choice
        pred_choice = extract_choice(generated_text)
        
        # 2. Extract Graph (Optional, for analysis)
        # We can save the whole text
        
        is_correct = (pred_choice == gt.strip('()')) # Handle (C) vs C
        if is_correct:
            correct_count += 1
            
        results.append({
            "idx": item.get('idx', i),
            "generated_text": generated_text,
            "pred_choice": pred_choice,
            "gt_choice": gt,
            "is_correct": is_correct
        })
    
    accuracy = correct_count / len(dataset)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()

