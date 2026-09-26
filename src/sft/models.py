import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


def find_all_linear_names(model):

    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_name = names[0] if len(names) == 1 else names[-1]
            if len(lora_name) == 1:
                lora_name = names[-2] + "." + lora_name
            lora_module_names.add(lora_name)

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def load_base_model_for_sft(
    base_model_path,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
    torch_dtype=torch.bfloat16,
    device_map=None,
    gradient_checkpointing=True,
):
    print(f"Loading base model from {base_model_path}...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
  
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
    
    print("Finding all linear layer names...")
    lora_modules = find_all_linear_names(model)
    print(f"Found {len(lora_modules)} unique linear layer names: {lora_modules}")
    
    print(f"Creating LoRA config with rank={lora_rank}, alpha={lora_alpha}...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    return model


def load_processor(base_model_path):
    print(f"Loading processor from {base_model_path}...")
    processor = Qwen2_5_VLProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    return processor
