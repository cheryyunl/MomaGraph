import argparse
import json
import os
import re
import base64
import io
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
# Import from the new prompts file
from prompts_direct import DIRECT_SYSTEM_PROMPT, DIRECT_QA_TEMPLATE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged HF model")
    parser.add_argument("--dataset_name", type=str, default="cheryyunl/MomaGraph-Bench", help="Dataset path")
    parser.add_argument("--output_file", type=str, default="eval_results_direct.jsonl")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()

def image_to_data_url(image):
    # Resize if too large
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
        
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

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

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    llm = LLM(
        model=args.model_path, 
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=16384, 
        limit_mm_per_prompt={"image": 4} 
    )
    
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train") 
    
    print(f"Processing {len(dataset)} samples...")
    
    # Prepare inputs
    inputs = []
    ground_truths = []
    idxs = []
    
    for item in dataset:
        image_url = image_to_data_url(item['image'])
        
        # Using simple direct prompt
        messages = [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}}, 
                {"type": "text", "text": DIRECT_QA_TEMPLATE.format(
                    task_instruction=item['task_instruction'],
                    question=item['question'],
                    options=", ".join(item['options']) if isinstance(item['options'], list) else item['options']
                )}
            ]}
        ]
        
        inputs.append(messages)
        ground_truths.append(item['correct_answer'])
        idxs.append(item.get('idx', len(idxs)))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=["<|endoftext|>", "<|im_end|>"]
    )
    
    print("Generating responses (Direct QA)...")
    
    outputs = llm.chat(
        messages=inputs,
        sampling_params=sampling_params
    )
    
    correct_count = 0
    results = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gt = ground_truths[i]
        
        pred_choice = extract_choice(generated_text)
        
        is_correct = (pred_choice == gt.strip('()'))
        if is_correct:
            correct_count += 1
            
        results.append({
            "idx": idxs[i],
            "generated_text": generated_text,
            "pred_choice": pred_choice,
            "gt_choice": gt,
            "is_correct": is_correct
        })
    
    accuracy = correct_count / len(dataset)
    print(f"Direct QA Accuracy: {accuracy:.4f}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()

