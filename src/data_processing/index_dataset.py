import json
import os
import argparse
import hashlib
import re
from pathlib import Path
from PIL import Image
import io
import pandas as pd

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def save_image_from_sample(raw_sample, images_dir, hash_value):
    try:
        images_dir = Path(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

        image_data = raw_sample.get('image')
        if not image_data:
            return None

        if isinstance(image_data, dict):
            if 'bytes' in image_data and image_data['bytes']:
                image_bytes = image_data['bytes']
                if isinstance(image_bytes, bytes):
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        format_ext = img.format.lower() if img.format else 'jpeg'
                        if format_ext == 'jpeg':
                            format_ext = 'jpg'

                        image_path = images_dir / f"{hash_value}.{format_ext}"
                        img.save(image_path)
                        return str(image_path)
                    except Exception as e:
                        print(f"Warning: Failed to save image from bytes: {e}")
        
        return None
    except Exception as e:
        print(f"Error saving image: {e}")
        return None


def create_dataset_index(
    dataset_root="./data/raw/RLAIF-V-Dataset",
    output_path="./data/index.jsonl",
    images_dir="./data/images",
    limit=None
):
    print(f"Creating dataset index from: {dataset_root}")
    print(f"Output index file: {output_path}")
    print(f"Images will be saved to: {images_dir}")

    parquet_dir = Path(dataset_root)
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_root}")

    parquet_files.sort()
    print(f"Found {len(parquet_files)} parquet files")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print("Processing samples...")
    seen_hashes = set()
    duplicate_count = 0
    image_save_success = 0
    image_save_failed = 0
    total_processed = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for parquet_file in parquet_files:
            print(f"Loading {parquet_file.name}...")
            df = pd.read_parquet(parquet_file)

            for _, row in df.iterrows():
                raw_sample = row.to_dict()

                # Generate hash
                chosen = raw_sample.get('chosen', '').strip()
                question = raw_sample.get('question', '').strip()

                if not chosen or not question:
                    continue
                hash_key = f"{norm(chosen)}|||{norm(question)}"
                hash_value = hashlib.md5(hash_key.encode()).hexdigest()

                # Check for duplicates
                if hash_value in seen_hashes:
                    duplicate_count += 1
                    continue

                seen_hashes.add(hash_value)

                saved_image_path = save_image_from_sample(raw_sample, images_dir, hash_value)
                if saved_image_path:
                    image_save_success += 1
                else:
                    image_save_failed += 1

                index_entry = {
                    'idx': total_processed,
                    'hash': hash_value,
                    'original_idx': raw_sample.get('idx', ''),
                    'question': question,
                    'chosen': chosen,
                    'rejected': raw_sample.get('rejected', ''),
                    'ds_name': raw_sample.get('ds_name', ''),
                    'origin_dataset': raw_sample.get('origin_dataset', ''),
                    'origin_split': raw_sample.get('origin_split', ''),
                    'image_path': saved_image_path if saved_image_path else raw_sample.get('image_path', ''),
                    'original_image_path': raw_sample.get('image_path', ''),
                    'has_image': bool(raw_sample.get('image', '')),
                    'image_saved': bool(saved_image_path),
                }

                f.write(json.dumps(index_entry, ensure_ascii=False) + '\n')
                total_processed += 1

                if total_processed % 1000 == 0:
                    print(f"Processed {total_processed} samples... (duplicates: {duplicate_count}, images saved: {image_save_success}, failed: {image_save_failed})")

                if limit is not None and total_processed >= limit:
                    break

            if limit is not None and total_processed >= limit:
                break

    unique_samples = len(seen_hashes)
    stats = {
        'total_samples_processed': total_processed,
        'unique_samples': unique_samples,
        'duplicate_samples': duplicate_count,
        'images_saved_successfully': image_save_success,
        'images_save_failed': image_save_failed,
        'output_file': output_path,
        'images_directory': images_dir
    }

    print("Index created successfully!")
    print(f"Total samples processed: {total_processed}")
    print(f"Unique samples: {unique_samples}")
    print(f"Duplicate samples removed: {duplicate_count}")
    print(f"Images saved successfully: {image_save_success}")
    print(f"Images save failed: {image_save_failed}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Create dataset index for RLAIF-V dataset")
    parser.add_argument("--dataset_root", type=str, default="./data/raw/RLAIF-V-Dataset")
    parser.add_argument("--output", type=str, default="./data/index.jsonl")
    parser.add_argument("--images_dir", type=str, default="./data/images")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    stats = create_dataset_index(
        dataset_root=args.dataset_root,
        output_path=args.output,
        images_dir=args.images_dir,
        limit=args.limit
    )

    print("Dataset indexing completed successfully!")
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    main()
