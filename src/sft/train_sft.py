import argparse
import os
import torch
from transformers import Trainer, TrainingArguments
from functools import partial
import json

from src.sft.models import load_base_model_for_sft, load_processor
from src.sft.data_utils import SFTDataset, collate_fn


class Qwen2_5_VLTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_grid_thw=inputs.get('image_grid_thw'),
            labels=inputs['labels'],
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description='SFT Training for Qwen2.5-VL using Trainer')

    # Model arguments
    parser.add_argument('--base_model_path', type=str, required=True)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading processor from {args.base_model_path}...")
    processor = load_processor(args.base_model_path)

    print(f"Loading model from {args.base_model_path}...")
    model = load_base_model_for_sft(
        base_model_path=args.base_model_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        torch_dtype=torch.bfloat16,
        device_map=None,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    print(f"Loading dataset from {args.data_path}...")
    dataset = SFTDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        processor=processor
    )
    print(f"Dataset size: {len(dataset)}")

    data_collator = partial(collate_fn, processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True, 
        dataloader_num_workers=4,
        remove_unused_columns=False,  
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        report_to="none", 
    )

    trainer = Qwen2_5_VLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"  Model: {args.base_model_path}")
    print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  Total epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * training_args.world_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  Output dir: {args.output_dir}")
    print(f"{'='*60}\n")

    print("Starting training...")
    trainer.train()

    print("\nSaving final model...")
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Training completed! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
