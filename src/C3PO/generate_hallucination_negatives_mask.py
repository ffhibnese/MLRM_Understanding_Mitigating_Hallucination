import json
import argparse
import os
from tqdm import tqdm
import re
import copy
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def mask_single_image_pil(image, mask_percentage, mask_method='random'):
    image_array = np.array(image) 
    H, W, C = image_array.shape
    mean_value = image_array.mean()
    masked_array = copy.deepcopy(image_array)
    
    if mask_method == 'random':
        total_pixels = H * W
        mask_pixels = int(total_pixels * mask_percentage)
        all_indices = np.arange(total_pixels)
        np.random.shuffle(all_indices)
        mask_indices = all_indices[:mask_pixels]
        flat_image = masked_array.reshape(-1, C)
        flat_image[mask_indices] = mean_value
        masked_array = flat_image.reshape(H, W, C)
        
    elif mask_method == 'blockwise':
        block_size = 14
        H_blocks = H // block_size
        W_blocks = W // block_size
        total_blocks = H_blocks * W_blocks
        mask_blocks = int(total_blocks * mask_percentage)
        all_block_indices = np.arange(total_blocks)
        np.random.shuffle(all_block_indices)
        mask_block_indices = all_block_indices[:mask_blocks]

        for idx in mask_block_indices:
            h = idx // W_blocks
            w = idx % W_blocks
            h_start = h * block_size
            h_end = h_start + block_size
            w_start = w * block_size
            w_end = w_start + block_size
            masked_array[h_start:h_end, w_start:w_end, :] = mean_value
    else:
        raise NotImplementedError(f"Mask method {mask_method} not implemented")

    masked_array = np.clip(masked_array, 0, 255).astype(np.uint8)
    masked_image = Image.fromarray(masked_array)
    
    return masked_image


def load_reference_model(base_model_path, lora_path):
    print(f"Loading base model from {base_model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side='left')
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = 'left'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    print("Model loaded successfully")
    
    return model, processor, tokenizer


def generate_masked_response_batch(model, processor, tokenizer, batch_questions, batch_image_paths, 
                                   mask_percentage, mask_method, max_new_tokens=15000):
    batch_images = []
    batch_texts = []
    
    for question, image_path in zip(batch_questions, batch_image_paths):
        original_image = Image.open(image_path).convert("RGB")
        masked_image = mask_single_image_pil(original_image, mask_percentage, mask_method)

        prompt_text = f"{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": masked_image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_images.append(masked_image)
        batch_texts.append(text)

    inputs = processor(
        text=batch_texts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    responses = []
    for i, output in enumerate(output_ids):
        generated_ids = output[inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        responses.append(response_text)
    
    return responses


def extract_reasoning_and_answer(text):
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)

    reasoning = think_match.group(1).strip() if think_match else None

    if answer_match:
        answer = answer_match.group(1).strip()
    elif think_match:
        answer = text.split('</think>')[1].strip()
    else:
        answer = ''

    return reasoning, answer


def main():
    parser = argparse.ArgumentParser(description="Generate mask-based hallucination negatives using reference model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to DPO data with normal negatives (output of generate_normal_negatives.py)")
    parser.add_argument("--base_model_path", type=str, default="your_target_model_path")
    parser.add_argument("--ref_lora_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default=".")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--mask_percentage", type=float, default=0.3)
    parser.add_argument("--mask_method", type=str, default="random", choices=["random", "blockwise"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    args = parser.parse_args()

    print(f"Loading DPO data from {args.input_path}...")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)

    if args.num_samples:
        dpo_data = dpo_data[:args.num_samples]

    print(f"Processing {len(dpo_data)} samples")
    print(f"Mask method: {args.mask_method}")
    print(f"Mask percentage: {args.mask_percentage}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    existing_hashes = set()
    existing_data = []

    if os.path.exists(args.output_path):
        print(f"Found existing output file: {args.output_path}")
        with open(args.output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_hashes = {item['hash'] for item in existing_data}
        print(f"Loaded {len(existing_data)} existing samples")

    samples_to_process = [item for item in dpo_data if item['hash'] not in existing_hashes]

    if not samples_to_process:
        print("All samples already have masked hallucination responses!")
    else:
        print(f"Need to generate masked responses for {len(samples_to_process)} new samples")

    if samples_to_process:
        model, processor, tokenizer = load_reference_model(args.base_model_path, args.ref_lora_path)

        print("Generating mask-based hallucination responses with reference model...")
        print(f"Processing in batches of {args.batch_size}")

        skip_cnt = 0

        for batch_start in tqdm(range(0, len(samples_to_process), args.batch_size), desc="Generating masked responses"):
            batch_samples = samples_to_process[batch_start:batch_start + args.batch_size]
            batch_questions = [item['question'] for item in batch_samples]
            batch_image_paths = [os.path.join(args.image_folder, item['image']) for item in batch_samples]

            masked_responses = generate_masked_response_batch(
                model, processor, tokenizer, batch_questions, batch_image_paths,
                args.mask_percentage, args.mask_method, args.max_new_tokens
            )

            for item, masked_response in zip(batch_samples, masked_responses):
                masked_reasoning, masked_answer = extract_reasoning_and_answer(masked_response)

                if not masked_reasoning or not masked_answer:
                    skip_cnt += 1
                    continue

                # Create new entry with all fields from input plus hallucination_negative
                masked_entry = item.copy()
                masked_entry['hallucination_negative'] = f"<think>{masked_reasoning}</think>\n\n<answer>{masked_answer}</answer>"
                masked_entry['hallucination_reasoning'] = masked_reasoning
                masked_entry['hallucination_answer'] = masked_answer
                masked_entry['mask_method'] = args.mask_method
                masked_entry['mask_percentage'] = args.mask_percentage

                existing_data.append(masked_entry)

            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

        print(f"\nGenerated {len(samples_to_process) - skip_cnt} masked responses")
        print(f"Skipped {skip_cnt} samples (extraction failed)")

    print(f"\n{'='*60}")
    print("C3PO DPO data (mask-based, ref model) generated successfully!")
    print(f"Total samples: {len(existing_data)}")
    print(f"Saved to: {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
