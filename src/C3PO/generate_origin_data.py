import json
import argparse
import os
import re
from pathlib import Path
from tqdm import tqdm
import random
from vllm import LLM, SamplingParams


def load_target_model(model_path, tensor_parallel_size=2):
    data_dir = Path("./data").resolve()

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=20480,
        limit_mm_per_prompt={"image": 10, "video": 0},
        allowed_local_media_path=str(data_dir),
    )
    return llm


def generate_mlrm_response_batch(llm, questions, image_paths):
    messages_batch = []
    for question, image_path in zip(questions, image_paths):
        prompt_text = f"{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."
        abs_path = Path(image_path).resolve()
        image_url = f"file://{abs_path}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        messages_batch.append(messages)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=15360, 
    )

    outputs = llm.chat(
        messages=messages_batch,
        sampling_params=sampling_params,
    )
    responses = [output.outputs[0].text.strip() for output in outputs]

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

def build_sft_dataset(index_jsonl, output_json, llm, num_samples=None, seed=42, batch_size=8):

    with open(index_jsonl, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]

    # Random sampling with seed
    if num_samples and num_samples < len(lines):
        random.seed(seed)
        lines = random.sample(lines, num_samples)
        print(f"Randomly sampled {num_samples} samples from {len(lines)} total samples (seed={seed})")

    all_samples = [json.loads(line) for line in lines]

    existing_samples = []
    existing_hashes = set()

    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                existing_samples = json.load(f)
                existing_hashes = {s['hash'] for s in existing_samples}
            print(f"Found {len(existing_samples)} existing samples")
        except Exception as e:
            print(f"Warning: Could not load {output_json}: {e}")
            existing_samples = []
            existing_hashes = set()

    samples_to_process = [s for s in all_samples if s.get('hash') not in existing_hashes]

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Already processed: {len(existing_hashes)}")
    print(f"To process: {len(samples_to_process)}")

    if len(samples_to_process) == 0:
        print("All samples already processed!")
        return len(existing_samples)

    skip_cnt = 0

    for batch_start in tqdm(range(0, len(samples_to_process), batch_size),
                            desc=f"Generating original CoT (batch_size={batch_size})"):
        batch_samples = samples_to_process[batch_start:batch_start + batch_size]
        questions = [s['question'] for s in batch_samples]
        image_paths = [s['image_path'] for s in batch_samples]
        mlrm_responses = generate_mlrm_response_batch(llm, questions, image_paths)

        for sample, mlrm_response in zip(batch_samples, mlrm_responses):
            reasoning, answer = extract_reasoning_and_answer(mlrm_response)

            if not reasoning or not answer:
                skip_cnt += 1
                continue

            sft_sample = {
                "hash": sample.get('hash'),
                "question": sample['question'],
                "image": sample['image_path'],
                "mlrm_full_response": mlrm_response,
                "original_reasoning": reasoning,
                "original_answer": answer,
            }

            existing_samples.append(sft_sample)

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(existing_samples, f, ensure_ascii=False, indent=2)

    print(f"\nSkipped {skip_cnt} samples due to extraction issues")
    print(f"Total samples generated: {len(existing_samples)}")

    return len(existing_samples)


def main():
    parser = argparse.ArgumentParser(description="Generate original CoT data (without compression)")
    parser.add_argument("--index", type=str, default="./data/index.jsonl")
    parser.add_argument("--target_model", type=str, default="your_target_model_path")
    parser.add_argument("--output", type=str, default="./outputs/sft_datasets/original_cot_data.json")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading target model with vLLM from: {args.target_model}")
    llm = load_target_model(args.target_model, tensor_parallel_size=args.tensor_parallel_size)
    print("Target model loaded")

    print(f"\n{'='*60}")
    print("Generating original CoT data (no compression)")
    print(f"Random sampling: {args.num_samples} samples (seed={args.seed})")
    print(f"{'='*60}")

    sample_count = build_sft_dataset(
        args.index, args.output, llm,
        args.num_samples, args.seed, args.batch_size
    )

    print(f"\n{'='*60}")
    print("Original CoT data generated successfully!")
    print(f"Total samples: {sample_count}")
    print(f"Saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
