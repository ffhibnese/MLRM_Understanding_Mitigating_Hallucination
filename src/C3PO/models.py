import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F

REGISTERED_BASE_MODELS = {}


def load_base_model(model_path, torch_dtype=torch.bfloat16, device_map=None, attn_implementation="flash_attention_2"):
    print(f"Loading base model from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    print(f"Model loaded: {model.__class__.__name__}")
    return model


def load_processor(model_path):
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'left'
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    else:
        processor.padding_side = 'left'
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
    print("Processor loaded")
    return processor


def create_lora_config(lora_rank=8, lora_alpha=16, lora_dropout=0.0, target_modules=None):
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return lora_config


def prepare_model_for_training(model, lora_config, gradient_checkpointing=True):
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_lora_model(base_model_path, lora_path, torch_dtype=torch.bfloat16, device_map=None):
    print(f"Loading base model from {base_model_path}...")
    base_model = load_base_model(base_model_path, torch_dtype=torch_dtype, device_map=device_map)
    print(f"Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch_dtype)
    print("LoRA model loaded")
    return model


def load_base_with_dual_lora(
    base_model_path,
    policy_lora_path,
    ref_lora_path=None,
    lora_rank=8,
    lora_alpha=16,
    torch_dtype=torch.bfloat16,
    device_map=None,
    gradient_checkpointing=True,
):
    global REGISTERED_BASE_MODELS

    if base_model_path in REGISTERED_BASE_MODELS:
        print(f"Reusing registered base model from {base_model_path}")
        return REGISTERED_BASE_MODELS[base_model_path]

    if ref_lora_path is None:
        ref_lora_path = policy_lora_path
        print("ref_lora_path not specified, using policy_lora_path as reference")
        print("This means ref will be the same as policy's initial state")

    print(f"Loading base model from {base_model_path}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(base_model.config, 'use_cache'):
            base_model.config.use_cache = False
        print("Gradient checkpointing enabled and use_cache disabled")

    print(f"Loading policy adapter from {policy_lora_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        policy_lora_path,
        adapter_name="lora_policy",
        is_trainable=True,
    )

    print(f"Loading reference adapter from {ref_lora_path}...")
    model.load_adapter(
        ref_lora_path,
        adapter_name="lora_ref_policy",
        is_trainable=False,
    )

    for name, param in model.named_parameters():
        if "lora_ref_policy" in name:
            param.requires_grad = False

    REGISTERED_BASE_MODELS[base_model_path] = model

    print("Dual-LoRA model loaded:")
    print(f"   - lora_policy (trainable): {policy_lora_path}")
    print(f"   - lora_ref_policy (frozen): {ref_lora_path}")
    return model


class PolicyModel(nn.Module):

    def __init__(self, model, tokenizer, adapter_name=None, response_len=896):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name  # "lora_policy" or "lora_ref_policy"
        self.response_len = response_len

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, query_len, temperature=1.0, mode="policy", **response_kwargs):

        if self.adapter_name is not None:
            self.model.set_adapter(self.adapter_name)

        self.model.config.use_cache = False

        response_keys = [k for k in response_kwargs.keys() if k.endswith('_input_ids')]

        if len(response_keys) == 0:
            with torch.set_grad_enabled(mode == "policy"):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                    output_hidden_states=False,
                )

            logits = outputs.logits[:, :-1, :] / temperature
            labels = input_ids[:, 1:].clone()
            attn_shifted = attention_mask[:, 1:]

            B, T = labels.size()

            pad_offset = torch.argmax((attn_shifted != 0).long(), dim=1)

            if not isinstance(query_len, torch.Tensor):
                query_len = torch.tensor([query_len] * B, dtype=torch.long, device=labels.device)
            elif query_len.dim() == 0:
                query_len = query_len.unsqueeze(0).expand(B)

            start_idx = pad_offset + (query_len - 1)
            arange_t = torch.arange(T, device=labels.device).unsqueeze(0).expand(B, T)
            valid_mask = arange_t >= start_idx.unsqueeze(1)

            labels[~valid_mask] = self.tokenizer.pad_token_id
            logprobs = self._compute_logprobs(logits, labels, ignore_index=self.tokenizer.pad_token_id)
            response_mask = ~labels.eq(self.tokenizer.pad_token_id)
            logprobs = logprobs * response_mask

            return logprobs

        else:

            all_input_ids = []
            all_attention_masks = []
            all_pixel_values = []
            all_image_grid_thw = []

            for key in response_keys:
                all_input_ids.append(response_kwargs[key])
                all_attention_masks.append(response_kwargs[key.replace('_input_ids', '_attention_mask')])
                all_pixel_values.append(pixel_values)
                all_image_grid_thw.append(image_grid_thw)

            max_len = max(ids.size(1) for ids in all_input_ids)

            padded_input_ids = []
            padded_attention_masks = []
            for ids, mask in zip(all_input_ids, all_attention_masks):
                if ids.size(1) < max_len:
                    pad_len = max_len - ids.size(1)
                    ids = F.pad(ids, (pad_len, 0), value=self.tokenizer.pad_token_id)
                    mask = F.pad(mask, (pad_len, 0), value=0)
                padded_input_ids.append(ids)
                padded_attention_masks.append(mask)

            merged_input_ids = torch.cat(padded_input_ids, dim=0)
            merged_attention_mask = torch.cat(padded_attention_masks, dim=0)
            merged_pixel_values = torch.cat(all_pixel_values, dim=0)
            merged_image_grid_thw = torch.cat(all_image_grid_thw, dim=0)

            with torch.set_grad_enabled(mode == "policy"):
                outputs = self.model(
                    input_ids=merged_input_ids,
                    attention_mask=merged_attention_mask,
                    pixel_values=merged_pixel_values,
                    image_grid_thw=merged_image_grid_thw,
                    use_cache=False,
                    output_hidden_states=False,
                )

            num_responses = len(response_keys)
            batch_size = all_input_ids[0].size(0)

            if not isinstance(query_len, torch.Tensor):
                query_len_tensor = torch.tensor([query_len] * batch_size, dtype=torch.long, device=merged_input_ids.device)
            elif query_len.dim() == 0:
                query_len_tensor = query_len.unsqueeze(0).expand(batch_size)
            else:
                query_len_tensor = query_len

            merged_query_len = query_len_tensor.repeat(num_responses)

            logits = outputs.logits[:, :-1, :] / temperature
            labels = merged_input_ids[:, 1:].clone()
            merged_attn_shifted = merged_attention_mask[:, 1:]

            B_merged, T = labels.size()

            pad_offset = torch.argmax((merged_attn_shifted != 0).long(), dim=1)

            start_idx = pad_offset + (merged_query_len - 1)
            arange_t = torch.arange(T, device=labels.device).unsqueeze(0).expand(B_merged, T)
            valid_mask = arange_t >= start_idx.unsqueeze(1)

            labels[~valid_mask] = self.tokenizer.pad_token_id

            logprobs = self._compute_logprobs(logits, labels, ignore_index=self.tokenizer.pad_token_id)
            response_mask = ~labels.eq(self.tokenizer.pad_token_id)
            logprobs = logprobs * response_mask

            batch_size = all_input_ids[0].size(0)
            result_dict = {}
            for i, key in enumerate(response_keys):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                result_key = key.replace('_input_ids', '_logprobs')
                result_dict[result_key] = logprobs[start_idx:end_idx]

            return result_dict
    
    def _compute_logprobs(self, logits, labels, ignore_index):
        logprobs = -F.cross_entropy(
            logits.permute(0, 2, 1),
            labels,
            reduction="none",
            ignore_index=ignore_index,
        )
        mask = (labels != ignore_index).float()
        logprobs = logprobs * mask
        return logprobs


def make_policy_and_ref_models(base_model_path, policy_lora_path=None, lora_config=None, response_len=896, torch_dtype=torch.bfloat16):

    processor = load_processor(base_model_path)
    print("Loading reference model...")
    ref_base_model = load_base_model(base_model_path, torch_dtype=torch_dtype)

    if policy_lora_path:
        print("Loading policy model with LoRA adapter...")
        policy_base_model = load_lora_model(base_model_path, policy_lora_path, torch_dtype=torch_dtype)
    else:
        print("Loading policy model from scratch...")
        policy_base_model = load_base_model(base_model_path, torch_dtype=torch_dtype)
        if lora_config:
            policy_base_model = prepare_model_for_training(policy_base_model, lora_config)

    policy_model = PolicyModel(policy_base_model, processor.tokenizer, response_len)
    ref_model = PolicyModel(ref_base_model, processor.tokenizer, response_len)

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    return policy_model, ref_model, processor
