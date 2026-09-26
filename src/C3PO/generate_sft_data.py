import json
import argparse
import os
from tqdm import tqdm
from src.tokenskip.compressor import TokenSkipCompressor

def load_index_data(index_jsonl):
    index_dict = {}
    if not os.path.exists(index_jsonl):
        return {}
    with open(index_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                index_dict[sample['hash']] = sample
    return index_dict

def generate_sft_data(sft_dataset_json, output_dir, compression_ratios, compressor, num_samples=None, index_jsonl=None):
    print(f"Loading dataset from {sft_dataset_json}...")
    with open(sft_dataset_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if num_samples:
        raw_data = raw_data[:num_samples]

    if index_jsonl:
        print(f"Loading index from {index_jsonl}...")
        index_dict = load_index_data(index_jsonl)
        if index_dict:
            valid_samples = []
            for s in raw_data:
                if s.get('hash') in index_dict:
                    valid_samples.append(s)
            print(f"Filtered {len(raw_data)} -> {len(valid_samples)} samples using index")
            data_to_process = valid_samples
        else:
            print("Index file empty or invalid, skipping filter.")
            data_to_process = raw_data
    else:
        data_to_process = raw_data

    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}

    for ratio in compression_ratios:
        print(f"\nProcessing compression ratio: {ratio}")
        sft_samples = []
        
        for sample in tqdm(data_to_process, desc=f"Compressing ratio {ratio}"):
            original_cot = sample.get('original_reasoning') or sample.get('original_cot')
            if not original_cot:
                continue
            
            original_answer = sample.get('original_answer') or sample.get('chosen') or sample.get('answer')
            if not original_answer:
                continue

            compressed = compressor.compress_cot(
                original_cot,
                compression_ratio=ratio
            )
            
            # Format: <think>...</think>\n\n<answer>...</answer>
            text = f"<think>{compressed['compressed_text']}</think>\n\n<answer>{original_answer}</answer>"
            
            sft_sample = {
                "hash": sample.get('hash'),
                "question": sample.get('question'),
                "image": sample.get('image'),
                "positive_response": text, 
                "compression_ratio": ratio,
                "original_tokens": compressed['original_tokens'],
                "compressed_tokens": compressed['compressed_tokens'],
                "weight": 1.0 
            }
            sft_samples.append(sft_sample)

        output_file = os.path.join(output_dir, f"sft_training_data_ratio_{ratio}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sft_samples, f, ensure_ascii=False, indent=2)
            
        stats[ratio] = len(sft_samples)
        print(f"Saved {len(sft_samples)} samples to {output_file}")

    return stats

def main():
    parser = argparse.ArgumentParser(description="Generate SFT data with TokenSkip compression")
    parser.add_argument("--sft_dataset", type=str, default="./outputs/sft_datasets/original_cot_data.json")
    parser.add_argument("--index", type=str, default="./data/index.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft_datasets")
    parser.add_argument("--compression_ratios", type=float, nargs='+', default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    parser.add_argument("--llmlingua_model", type=str, default="your_llmlingua_model_path")
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    print(f"Loading TokenSkip compressor from {args.llmlingua_model}...")
    compressor = TokenSkipCompressor(model_path=args.llmlingua_model)

    generate_sft_data(
        args.sft_dataset,
        args.output_dir,
        args.compression_ratios,
        compressor,
        args.num_samples,
        args.index
    )

if __name__ == "__main__":
    main()
