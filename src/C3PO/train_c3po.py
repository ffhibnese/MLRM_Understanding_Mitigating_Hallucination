import argparse
from pathlib import Path
import json
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin

from .models import load_base_with_dual_lora, load_processor, PolicyModel
from .data_utils import make_c3po_data_module
from .dpo_trainer import C3POTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="C3PO training")
    
    # Model paths
    parser.add_argument("--base_model", type=str, default="your_target_model_path")
    parser.add_argument("--policy_lora_path", type=str, default="your_policy_lora_path")
    parser.add_argument("--ref_lora_path", type=str, default="your_ref_lora_path")
    parser.add_argument("--output_dir", type=str, default="./outputs/c3po_models")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # Data paths
    parser.add_argument("--data_path_instruction", type=str, default=None)
    parser.add_argument("--data_path_image", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="./data")
    
    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    
    # DPO config
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambda_anc", type=float, default=1.0)
    parser.add_argument("--anchor_value", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    # DPO Loss control
    parser.add_argument("--lambda_dpo", type=float, default=1.0)
    
    # Training config
    parser.add_argument("--total_epochs", type=int, default=4)
    parser.add_argument("--rollout_batch_size", type=int, default=128)
    parser.add_argument("--rollout_per_device_batch_size", type=int, default=8)
    parser.add_argument("--step_batch_size", type=int, default=16)
    parser.add_argument("--step_per_device_batch_size", type=int, default=4)
    parser.add_argument("--noptepochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default=None, choices=[None, "linear", "cosine", "constant"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    
    # Logging and saving
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--query_len", type=int, default=256)
    parser.add_argument("--response_len", type=int, default=1024)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()

    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=1,
        sync_with_dataloader=False,
    )
    accelerator = Accelerator(
        mixed_precision='bf16' if args.bf16 else 'no',
        gradient_accumulation_plugin=gradient_accumulation_plugin,
    )

    world_size = accelerator.num_processes

    if args.total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if args.noptepochs <= 0:
        raise ValueError("noptepochs must be positive")
    if args.rollout_per_device_batch_size <= 0 or args.step_per_device_batch_size <= 0:
        raise ValueError("per-device batch sizes must be positive")
    if args.rollout_batch_size < args.rollout_per_device_batch_size * world_size:
        raise ValueError("rollout_batch_size is smaller than the distributed per-device batch size")
    if args.rollout_batch_size % (args.rollout_per_device_batch_size * world_size) != 0:
        raise ValueError("rollout_batch_size must be divisible by rollout_per_device_batch_size * world_size")
    if args.step_batch_size < args.step_per_device_batch_size * world_size:
        raise ValueError("step_batch_size is smaller than the distributed per-device batch size")
    if args.step_batch_size % (args.step_per_device_batch_size * world_size) != 0:
        raise ValueError("step_batch_size must be divisible by step_per_device_batch_size * world_size")
    if args.rollout_batch_size % args.step_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by step_batch_size")
    if args.logging_steps <= 0:
        raise ValueError("logging_steps must be positive")

    args.rollout_accumulation_steps = args.rollout_batch_size // args.rollout_per_device_batch_size // world_size
    args.gradient_accumulation_steps = args.step_batch_size // args.step_per_device_batch_size // world_size

    accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*60)
    print("C3PO Training")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Policy LoRA: {args.policy_lora_path}")
    print(f"Reference LoRA: {args.ref_lora_path}")
    if args.data_path_instruction:
        print(f"Instruction hallucination data: {args.data_path_instruction}")
    if args.data_path_image:
        print(f"Image hallucination data: {args.data_path_image}")
    print(f"Output dir: {args.output_dir}")
    print(f"Beta: {args.beta}")
    print(f"Lambda Anc: {args.lambda_anc}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Total epochs: {args.total_epochs}")
    print("\nCoT Loss configuration:")
    print(f"  Lambda DPO: {args.lambda_dpo}")
    print("\nBatch size configuration:")
    print(f"  Rollout batch size: {args.rollout_batch_size}")
    print(f"  Rollout per-device batch size: {args.rollout_per_device_batch_size}")
    print(f"  Rollout accumulation steps: {args.rollout_accumulation_steps}")
    print(f"  Step batch size: {args.step_batch_size}")
    print(f"  Step per-device batch size: {args.step_per_device_batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Noptepochs: {args.noptepochs}")
    print(f"  World size: {world_size}")
    print("="*60)

    print("\nLoading models...")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    processor = load_processor(args.base_model)

    print("Loading base model with dual LoRA adapters...")
    policy_lora_path = args.resume_from_checkpoint or args.policy_lora_path
    shared_model = load_base_with_dual_lora(
        base_model_path=args.base_model,
        policy_lora_path=policy_lora_path,
        ref_lora_path=args.ref_lora_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        torch_dtype=torch_dtype,
        device_map=None,
        gradient_checkpointing=True,
    )

    policy_model = PolicyModel(
        shared_model,
        processor.tokenizer,
        adapter_name="lora_policy",
        response_len=args.response_len,
    )

    ref_model = PolicyModel(
        shared_model,
        processor.tokenizer,
        adapter_name="lora_ref_policy",
        response_len=args.response_len,
    )

    policy_model = accelerator.prepare(policy_model)

    print("\nLoading data...")
    data_module = make_c3po_data_module(
        processor=processor,
        data_path_instruction=args.data_path_instruction,
        data_path_image=args.data_path_image,
        image_folder=args.image_folder,
        max_length=args.max_length,
        query_len=args.query_len,
        response_len=args.response_len,
    )

    print(f"Train dataset size: {len(data_module['train_dataset'])}")

    trainer = C3POTrainer(
        args=args,
        policy=policy_model,
        ref_policy=ref_model,
        accelerator=accelerator,
        processor=processor,
        **data_module,
    )
    
    print("\nStarting training...")
    trainer.train(resume_training_ckpt=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
