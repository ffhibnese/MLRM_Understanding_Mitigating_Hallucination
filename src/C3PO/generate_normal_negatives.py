import json
import argparse
import os
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image


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


def generate_normal_response_batch(model, processor, tokenizer, batch_questions, batch_image_paths, max_new_tokens=15000):
    batch_images = []
    batch_texts = []

    for question, image_path in zip(batch_questions, batch_image_paths):
        image = Image.open(image_path).convert("RGB")
        prompt_text = f"{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_images.append(image)
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
    parser = argparse.ArgumentParser(description="Generate normal negatives using reference model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to base DPO data (output of generate_base_dpo_data.py)")
    parser.add_argument("--base_model_path", type=str, default="your_target_model_path")
    parser.add_argument("--ref_lora_path", type=str, required=True, help="Path to reference model LoRA adapter")
    parser.add_argument("--image_folder", type=str, default=".")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    args = parser.parse_args()

    print(f"Loading DPO data from {args.input_path}...")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)

    if args.num_samples:
        dpo_data = dpo_data[:args.num_samples]

    print(f"Processing {len(dpo_data)} samples")

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
        print("All samples already have normal negative responses!")
    else:
        print(f"Need to generate normal negative responses for {len(samples_to_process)} new samples")

    if samples_to_process:
        model, processor, tokenizer = load_reference_model(args.base_model_path, args.ref_lora_path)

        print("Generating normal negative responses with reference model...")
        print(f"Processing in batches of {args.batch_size}")

        skip_cnt = 0

        for batch_start in tqdm(range(0, len(samples_to_process), args.batch_size), desc="Generating normal negatives"):
            batch_samples = samples_to_process[batch_start:batch_start + args.batch_size]
            batch_questions = [item['question'] for item in batch_samples]
            batch_image_paths = [os.path.join(args.image_folder, item['image']) for item in batch_samples]

            normal_responses = generate_normal_response_batch(
                model, processor, tokenizer, batch_questions, batch_image_paths, args.max_new_tokens
            )

            for item, normal_response in zip(batch_samples, normal_responses):
                normal_reasoning, normal_answer = extract_reasoning_and_answer(normal_response)

                if not normal_reasoning or not normal_answer:
                    skip_cnt += 1
                    updated_entry = item.copy()
                else:
                    # Update negative_response with new generation
                    updated_entry = item.copy()
                    updated_entry['negative_response'] = normal_response

                existing_data.append(updated_entry)

            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

        print(f"\nGenerated {len(samples_to_process) - skip_cnt} normal negative responses")
        print(f"Skipped {skip_cnt} samples (extraction failed, kept original negative_response)")

    print(f"\n{'='*60}")
    print("DPO data with updated normal negatives generated successfully!")
    print(f"Total samples: {len(existing_data)}")
    print(f"Saved to: {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
