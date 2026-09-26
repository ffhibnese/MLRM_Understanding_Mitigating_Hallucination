import os
import re
import torch
import torch.nn.functional as F

from .rl_trainer import RLTrainer

OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"


class C3POTrainer(RLTrainer):

    def __init__(self, args, train_dataset, data_collator, policy, ref_policy, accelerator, processor=None, **kwargs):
        super().__init__(
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            policy=policy,
            ref_policy=ref_policy,
            accelerator=accelerator,
            **kwargs
        )
        self.processor = processor
        self.beta = args.beta
        self.lambda_anc = args.lambda_anc
        self.anchor_value = args.anchor_value
        self.temperature = args.temperature
        self.lambda_dpo = args.lambda_dpo

        tokenizer = processor.tokenizer
        self.think_start_ids = tokenizer("<think", add_special_tokens=False).input_ids
        end_patterns = [
            "</think",
            "</think>",
            "\n</think",
            "\n</think>",
            " </think",
            " </think>",
            "</think>\n",
            "</think>\n\n",
            "</think>\n\n<answer",
            ".</think",
            ".</think>",
            ".</think>\n",
            ".</think>\n\n",
            ".</think>\n\n<answer",
        ]
        self.think_end_patterns = []
        for pat in end_patterns:
            ids = tokenizer(pat, add_special_tokens=False).input_ids
            if ids and ids not in self.think_end_patterns:
                self.think_end_patterns.append(ids)

    def _find_subseq(self, seq, pat, start=0):
        n, m = len(seq), len(pat)
        if m == 0 or n < m:
            return None
        for i in range(start, n - m + 1):
            if seq[i:i+m] == pat:
                return i
        return None

    def _get_cot_mask(self, input_ids, attention_mask, query_len):
        B, T = input_ids.shape
        device = input_ids.device

        if not isinstance(query_len, torch.Tensor):
            query_len = torch.tensor([query_len] * B, dtype=torch.long, device=device)
        elif query_len.dim() == 0:
            query_len = query_len.unsqueeze(0).expand(B)

        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        pad_offset = torch.argmax((attention_mask != 0).long(), dim=1)

        for i in range(B):
            start_search = (pad_offset[i] + query_len[i]).item()
            seq = input_ids[i].tolist()

            s = self._find_subseq(seq, self.think_start_ids, start=start_search)

            if s is None:
                continue

            end_search = s + len(self.think_start_ids)
            e = None
            matched_pat = None
            for pat in self.think_end_patterns:
                pos = self._find_subseq(seq, pat, start=end_search)
                if pos is not None and (e is None or pos < e):
                    e = pos
                    matched_pat = pat

            if e is None:
                continue

            mask[i, s : e + len(matched_pat)] = 1.0

        return mask

    @torch.inference_mode()
    def rollout(self, queries_batches):
        self.policy.eval()
        self.ref_policy.eval()
        rollouts = []

        for batch in queries_batches:
            # Positive samples
            positive_input_ids = batch['positive_input_ids'].to(self.accelerator.device)
            positive_attention_mask = batch['positive_attention_mask'].to(self.accelerator.device)
            positive_pixel_values = batch['positive_pixel_values'].to(self.accelerator.device)
            positive_image_grid_thw = batch['positive_image_grid_thw'].to(self.accelerator.device) if batch['positive_image_grid_thw'] is not None else None

            # Negative samples (for L_DPO)
            negative_input_ids = batch['negative_input_ids'].to(self.accelerator.device)
            negative_attention_mask = batch['negative_attention_mask'].to(self.accelerator.device)
            negative_pixel_values = batch['negative_pixel_values'].to(self.accelerator.device)
            negative_image_grid_thw = batch['negative_image_grid_thw'].to(self.accelerator.device) if batch['negative_image_grid_thw'] is not None else None

            query_len = batch['query_len'].to(self.accelerator.device)

            # Reference logprobs for positive
            ref_positive_logprobs = self.ref_policy(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask,
                pixel_values=positive_pixel_values,
                image_grid_thw=positive_image_grid_thw,
                query_len=query_len,
                temperature=self.temperature,
                mode="reference",
            )

            # Reference logprobs for negative (L_DPO)
            ref_negative_logprobs = self.ref_policy(
                input_ids=negative_input_ids,
                attention_mask=negative_attention_mask,
                pixel_values=negative_pixel_values,
                image_grid_thw=negative_image_grid_thw,
                query_len=query_len,
                temperature=self.temperature,
                mode="reference",
            )

            ref_positive_total = ref_positive_logprobs.sum(dim=-1)
            ref_negative_total = ref_negative_logprobs.sum(dim=-1)

            rollout_batch = {
                'positive_input_ids': positive_input_ids,
                'positive_attention_mask': positive_attention_mask,
                'positive_pixel_values': positive_pixel_values,
                'positive_image_grid_thw': positive_image_grid_thw,
                'negative_input_ids': negative_input_ids,
                'negative_attention_mask': negative_attention_mask,
                'negative_pixel_values': negative_pixel_values,
                'negative_image_grid_thw': negative_image_grid_thw,
                'query_len': query_len,
                'ref_positive_logprobs': ref_positive_total,
                'ref_negative_logprobs': ref_negative_total,
                'ref_positive_total': ref_positive_total,
            }

            # CoT Loss
            # Positive mask
            cot_mask_positive = self._get_cot_mask(positive_input_ids, positive_attention_mask, query_len)
            cot_mask_positive_shifted = cot_mask_positive[:, 1:]
            ref_positive_cot_sum = (ref_positive_logprobs * cot_mask_positive_shifted).sum(dim=-1)

            # Negative mask
            cot_mask_negative = self._get_cot_mask(negative_input_ids, negative_attention_mask, query_len)
            cot_mask_negative_shifted = cot_mask_negative[:, 1:]
            ref_negative_cot_sum = (ref_negative_logprobs * cot_mask_negative_shifted).sum(dim=-1)

            rollout_batch['cot_mask_positive'] = cot_mask_positive
            rollout_batch['cot_mask_negative'] = cot_mask_negative
            rollout_batch['ref_positive_cot_sum'] = ref_positive_cot_sum
            rollout_batch['ref_negative_cot_sum'] = ref_negative_cot_sum

            need_hallucination_instruction = 'hallucination_instruction_input_ids' in batch
            need_hallucination_image = 'hallucination_image_input_ids' in batch

            if need_hallucination_instruction:
                hallucination_instruction_input_ids = batch['hallucination_instruction_input_ids'].to(self.accelerator.device)
                hallucination_instruction_attention_mask = batch['hallucination_instruction_attention_mask'].to(self.accelerator.device)
                hallucination_instruction_pixel_values = batch['hallucination_instruction_pixel_values'].to(self.accelerator.device)
                hallucination_instruction_image_grid_thw = batch['hallucination_instruction_image_grid_thw'].to(self.accelerator.device) if batch['hallucination_instruction_image_grid_thw'] is not None else None

                ref_hallucination_instruction_logprobs = self.ref_policy(
                    input_ids=hallucination_instruction_input_ids,
                    attention_mask=hallucination_instruction_attention_mask,
                    pixel_values=hallucination_instruction_pixel_values,
                    image_grid_thw=hallucination_instruction_image_grid_thw,
                    query_len=query_len,
                    temperature=self.temperature,
                    mode="reference",
                )
                ref_hallucination_instruction_total = ref_hallucination_instruction_logprobs.sum(dim=-1)

                rollout_batch['hallucination_instruction_input_ids'] = hallucination_instruction_input_ids
                rollout_batch['hallucination_instruction_attention_mask'] = hallucination_instruction_attention_mask
                rollout_batch['hallucination_instruction_pixel_values'] = hallucination_instruction_pixel_values
                rollout_batch['hallucination_instruction_image_grid_thw'] = hallucination_instruction_image_grid_thw
                rollout_batch['ref_hallucination_instruction_total'] = ref_hallucination_instruction_total
                rollout_batch['hallucination_instruction_mask'] = batch.get('hallucination_instruction_mask', None)

                cot_mask_hinstr = self._get_cot_mask(hallucination_instruction_input_ids, hallucination_instruction_attention_mask, query_len)
                cot_mask_hinstr_shifted = cot_mask_hinstr[:, 1:]
                ref_hallucination_instruction_cot_sum = (ref_hallucination_instruction_logprobs * cot_mask_hinstr_shifted).sum(dim=-1)
                rollout_batch['cot_mask_hallucination_instruction'] = cot_mask_hinstr
                rollout_batch['ref_hallucination_instruction_cot_sum'] = ref_hallucination_instruction_cot_sum

            if need_hallucination_image:
                hallucination_image_input_ids = batch['hallucination_image_input_ids'].to(self.accelerator.device)
                hallucination_image_attention_mask = batch['hallucination_image_attention_mask'].to(self.accelerator.device)
                hallucination_image_pixel_values = batch['hallucination_image_pixel_values'].to(self.accelerator.device)
                hallucination_image_image_grid_thw = batch['hallucination_image_image_grid_thw'].to(self.accelerator.device) if batch['hallucination_image_image_grid_thw'] is not None else None

                ref_hallucination_image_logprobs = self.ref_policy(
                    input_ids=hallucination_image_input_ids,
                    attention_mask=hallucination_image_attention_mask,
                    pixel_values=hallucination_image_pixel_values,
                    image_grid_thw=hallucination_image_image_grid_thw,
                    query_len=query_len,
                    temperature=self.temperature,
                    mode="reference",
                )
                ref_hallucination_image_total = ref_hallucination_image_logprobs.sum(dim=-1)

                rollout_batch['hallucination_image_input_ids'] = hallucination_image_input_ids
                rollout_batch['hallucination_image_attention_mask'] = hallucination_image_attention_mask
                rollout_batch['hallucination_image_pixel_values'] = hallucination_image_pixel_values
                rollout_batch['hallucination_image_image_grid_thw'] = hallucination_image_image_grid_thw
                rollout_batch['ref_hallucination_image_total'] = ref_hallucination_image_total
                rollout_batch['hallucination_image_mask'] = batch.get('hallucination_image_mask', None)

                # CoT sum for image hallucination
                cot_mask_himg = self._get_cot_mask(hallucination_image_input_ids, hallucination_image_attention_mask, query_len)
                cot_mask_himg_shifted = cot_mask_himg[:, 1:]
                ref_hallucination_image_cot_sum = (ref_hallucination_image_logprobs * cot_mask_himg_shifted).sum(dim=-1)
                rollout_batch['cot_mask_hallucination_image'] = cot_mask_himg
                rollout_batch['ref_hallucination_image_cot_sum'] = ref_hallucination_image_cot_sum

            rollouts_batch_cpu = {
                k: v.cpu() if torch.is_tensor(v) and v is not None else v
                for k, v in rollout_batch.items()
            }
            rollouts.append(rollouts_batch_cpu)

        if len(rollouts) == 0:
            return {}

        all_keys = set().union(*[r.keys() for r in rollouts])

        merged_rollouts = {}
        for key in all_keys:
            vals = [r.get(key, None) for r in rollouts]

            proto = next((v for v in vals if torch.is_tensor(v)), None)

            if proto is not None:

                if 'pixel_values' in key or 'image_grid_thw' in key:
                    merged_rollouts[key] = vals
                else:
                    parts = []
                    for v, r in zip(vals, rollouts):
                        if torch.is_tensor(v):
                            parts.append(v)
                        else:
                            bsz = r['positive_input_ids'].shape[0]
                            filler = torch.zeros((bsz, *proto.shape[1:]), dtype=proto.dtype, device=proto.device)
                            parts.append(filler)
                    merged_rollouts[key] = torch.cat(parts, dim=0)
            elif all(v is None for v in vals):
                pass

        return merged_rollouts

    def compute_policy_loss(self, rollouts_batch):

        self.policy.train()

        device = self.accelerator.device
        stats = {}

        positive_input_ids = rollouts_batch['positive_input_ids'].to(device)
        positive_attention_mask = rollouts_batch['positive_attention_mask'].to(device)
        positive_pixel_values = rollouts_batch['positive_pixel_values'].to(device)
        positive_image_grid_thw = rollouts_batch['positive_image_grid_thw'].to(device) if rollouts_batch['positive_image_grid_thw'] is not None else None

        negative_input_ids = rollouts_batch['negative_input_ids'].to(device)
        negative_attention_mask = rollouts_batch['negative_attention_mask'].to(device)
        negative_pixel_values = rollouts_batch['negative_pixel_values'].to(device)
        negative_image_grid_thw = rollouts_batch['negative_image_grid_thw'].to(device) if rollouts_batch['negative_image_grid_thw'] is not None else None

        query_len = rollouts_batch['query_len'].to(device)
        ref_positive_logprobs = rollouts_batch['ref_positive_logprobs'].to(device)
        ref_negative_logprobs = rollouts_batch['ref_negative_logprobs'].to(device)
        ref_positive_total = rollouts_batch['ref_positive_total'].to(device)

        # Policy logprobs for positive
        policy_positive_logprobs = self.policy(
            input_ids=positive_input_ids,
            attention_mask=positive_attention_mask,
            pixel_values=positive_pixel_values,
            image_grid_thw=positive_image_grid_thw,
            query_len=query_len,
            temperature=self.temperature,
            mode="policy",
        )

        # Policy logprobs for negative (L_DPO)
        policy_negative_logprobs = self.policy(
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
            pixel_values=negative_pixel_values,
            image_grid_thw=negative_image_grid_thw,
            query_len=query_len,
            temperature=self.temperature,
            mode="policy",
        )

        policy_positive_total = policy_positive_logprobs.sum(dim=-1)
        policy_negative_total = policy_negative_logprobs.sum(dim=-1)

        # L_gen
        chosen_logratios_gen = policy_positive_total - ref_positive_logprobs
        rejected_logratios_gen = policy_negative_total - ref_negative_logprobs

        logits_gen = chosen_logratios_gen - rejected_logratios_gen
        losses_gen = -F.logsigmoid(self.beta * logits_gen)
        loss_gen = losses_gen.mean()

        # Instruction-based hallucination (DPO Instr)
        loss_lif_instruction = torch.tensor(0.0, device=device)
        policy_hallucination_instruction_cot_sum_cached = None
        
        if 'hallucination_instruction_input_ids' in rollouts_batch:
            hallucination_instruction_input_ids = rollouts_batch['hallucination_instruction_input_ids'].to(device)
            hallucination_instruction_attention_mask = rollouts_batch['hallucination_instruction_attention_mask'].to(device)
            hallucination_instruction_pixel_values = rollouts_batch['hallucination_instruction_pixel_values'].to(device)
            hallucination_instruction_image_grid_thw = rollouts_batch['hallucination_instruction_image_grid_thw'].to(device) if rollouts_batch['hallucination_instruction_image_grid_thw'] is not None else None

            ref_hallucination_instruction_total = rollouts_batch['ref_hallucination_instruction_total'].to(device)

            policy_hallucination_instruction_logprobs = self.policy(
                input_ids=hallucination_instruction_input_ids,
                attention_mask=hallucination_instruction_attention_mask,
                pixel_values=hallucination_instruction_pixel_values,
                image_grid_thw=hallucination_instruction_image_grid_thw,
                query_len=query_len,
                temperature=self.temperature,
                mode="policy",
            )
            policy_hallucination_instruction_total = policy_hallucination_instruction_logprobs.sum(dim=-1)

            if 'cot_mask_hallucination_instruction' in rollouts_batch:
                cot_mask_hinstr = rollouts_batch['cot_mask_hallucination_instruction'].to(device)
                cot_mask_hinstr_shifted = cot_mask_hinstr[:, 1:]
                policy_hallucination_instruction_cot_sum_cached = (policy_hallucination_instruction_logprobs * cot_mask_hinstr_shifted).sum(dim=-1)

            chosen_logratios_lif_instruction = policy_positive_total - ref_positive_total
            rejected_logratios_lif_instruction = policy_hallucination_instruction_total - ref_hallucination_instruction_total

            logits_lif_instruction = chosen_logratios_lif_instruction - rejected_logratios_lif_instruction
            losses_lif_instruction = -F.logsigmoid(self.beta * logits_lif_instruction)

            instr_mask = rollouts_batch.get('hallucination_instruction_mask', None)
            if instr_mask is not None:
                instr_mask = instr_mask.to(device).float()
                denom = instr_mask.sum().clamp(min=1.0)
                loss_lif_instruction = (losses_lif_instruction * instr_mask).sum() / denom
            else:
                loss_lif_instruction = losses_lif_instruction.mean()

            stats['logprobs/policy_hallucination_instruction'] = policy_hallucination_instruction_total.mean().detach()
            stats['logprobs/ref_hallucination_instruction'] = ref_hallucination_instruction_total.mean().detach()
            stats['logratios/chosen_lif_instruction'] = chosen_logratios_lif_instruction.mean().detach()
            stats['logratios/rejected_lif_instruction'] = rejected_logratios_lif_instruction.mean().detach()
            stats['rewards/margin_lif_instruction'] = (self.beta * logits_lif_instruction).mean().detach()

        # Image-based hallucination (DPO Image)
        loss_lif_image = torch.tensor(0.0, device=device)
        policy_hallucination_image_cot_sum_cached = None
        
        if 'hallucination_image_input_ids' in rollouts_batch:
            if rollouts_batch['hallucination_image_pixel_values'] is None:
                print("Warning: Skipping a batch because hallucination_image_pixel_values is None.")
            else:
                hallucination_image_input_ids = rollouts_batch['hallucination_image_input_ids'].to(device)
                hallucination_image_attention_mask = rollouts_batch['hallucination_image_attention_mask'].to(device)
                hallucination_image_pixel_values = rollouts_batch['hallucination_image_pixel_values'].to(device)
                hallucination_image_image_grid_thw = rollouts_batch['hallucination_image_image_grid_thw'].to(device) if rollouts_batch['hallucination_image_image_grid_thw'] is not None else None

                ref_hallucination_image_total = rollouts_batch['ref_hallucination_image_total'].to(device)

                policy_hallucination_image_logprobs = self.policy(
                    input_ids=hallucination_image_input_ids,
                    attention_mask=hallucination_image_attention_mask,
                    pixel_values=hallucination_image_pixel_values,
                    image_grid_thw=hallucination_image_image_grid_thw,
                    query_len=query_len,
                    temperature=self.temperature,
                    mode="policy",
                )
                policy_hallucination_image_total = policy_hallucination_image_logprobs.sum(dim=-1)

                if 'cot_mask_hallucination_image' in rollouts_batch:
                    cot_mask_himg = rollouts_batch['cot_mask_hallucination_image'].to(device)
                    cot_mask_himg_shifted = cot_mask_himg[:, 1:]
                    policy_hallucination_image_cot_sum_cached = (policy_hallucination_image_logprobs * cot_mask_himg_shifted).sum(dim=-1)

                chosen_logratios_lif_image = policy_positive_total - ref_positive_total
                rejected_logratios_lif_image = policy_hallucination_image_total - ref_hallucination_image_total

                logits_lif_image = chosen_logratios_lif_image - rejected_logratios_lif_image
                losses_lif_image = -F.logsigmoid(self.beta * logits_lif_image)

                img_mask = rollouts_batch.get('hallucination_image_mask', None)
                if img_mask is not None:
                    img_mask = img_mask.to(device).float()
                    denom = img_mask.sum().clamp(min=1.0)
                    loss_lif_image = (losses_lif_image * img_mask).sum() / denom
                else:
                    loss_lif_image = losses_lif_image.mean()

                stats['logprobs/policy_hallucination_image'] = policy_hallucination_image_total.mean().detach()
                stats['logprobs/ref_hallucination_image'] = ref_hallucination_image_total.mean().detach()
                stats['logratios/chosen_lif_image'] = chosen_logratios_lif_image.mean().detach()
                stats['logratios/rejected_lif_image'] = rejected_logratios_lif_image.mean().detach()
                stats['rewards/margin_lif_image'] = (self.beta * logits_lif_image).mean().detach()

        # L_Anc_DPO (Full Anchor)
        chosen_logratios_anchor = policy_positive_total - ref_positive_total
        anchor_losses = -F.logsigmoid(self.beta * (chosen_logratios_anchor - self.anchor_value))
        loss_anchor = anchor_losses.mean() # This is the full output anchor loss

        # CoT Loss Calculation
        loss_cot_dpo_base = torch.tensor(0.0, device=device)
        loss_cot_dpo_instr = torch.tensor(0.0, device=device)
        loss_cot_dpo_img = torch.tensor(0.0, device=device)
        loss_cot_anchor = torch.tensor(0.0, device=device)

        if 'cot_mask_positive' in rollouts_batch:
            cot_mask_positive = rollouts_batch['cot_mask_positive'].to(device)
            cot_mask_negative = rollouts_batch['cot_mask_negative'].to(device)
            cot_mask_positive_shifted = cot_mask_positive[:, 1:]
            cot_mask_negative_shifted = cot_mask_negative[:, 1:]

            ref_positive_cot_sum = rollouts_batch['ref_positive_cot_sum'].to(device)
            ref_negative_cot_sum = rollouts_batch['ref_negative_cot_sum'].to(device)

            policy_positive_cot_sum = (policy_positive_logprobs * cot_mask_positive_shifted).sum(dim=-1)
            policy_negative_cot_sum = (policy_negative_logprobs * cot_mask_negative_shifted).sum(dim=-1)

            chosen_cot_logratios = policy_positive_cot_sum - ref_positive_cot_sum
            rejected_cot_logratios = policy_negative_cot_sum - ref_negative_cot_sum

            cot_dpo_logits = chosen_cot_logratios - rejected_cot_logratios
            cot_dpo_losses = -F.logsigmoid(self.beta * cot_dpo_logits)
            loss_cot_dpo_base = cot_dpo_losses.mean()

            # L_RE (CoT Instr)
            if 'ref_hallucination_instruction_cot_sum' in rollouts_batch:
                ref_hallucination_instruction_cot_sum = rollouts_batch['ref_hallucination_instruction_cot_sum'].to(device)

                if policy_hallucination_instruction_cot_sum_cached is not None:
                    rejected_cot_logratios_instr = policy_hallucination_instruction_cot_sum_cached - ref_hallucination_instruction_cot_sum
                    cot_dpo_logits_instr = chosen_cot_logratios - rejected_cot_logratios_instr
                    cot_dpo_losses_instr = -F.logsigmoid(self.beta * cot_dpo_logits_instr)

                    instr_mask = rollouts_batch.get('hallucination_instruction_mask', None)
                    if instr_mask is not None:
                        instr_mask = instr_mask.to(device).float()
                        denom = instr_mask.sum().clamp(min=1.0)
                        loss_cot_dpo_instr = (cot_dpo_losses_instr * instr_mask).sum() / denom
                    else:
                        loss_cot_dpo_instr = cot_dpo_losses_instr.mean()

            # L_RE (CoT Img)
            if 'ref_hallucination_image_cot_sum' in rollouts_batch:
                ref_hallucination_image_cot_sum = rollouts_batch['ref_hallucination_image_cot_sum'].to(device)

                if policy_hallucination_image_cot_sum_cached is not None:
                    rejected_cot_logratios_img = policy_hallucination_image_cot_sum_cached - ref_hallucination_image_cot_sum
                    cot_dpo_logits_img = chosen_cot_logratios - rejected_cot_logratios_img
                    cot_dpo_losses_img = -F.logsigmoid(self.beta * cot_dpo_logits_img)

                    img_mask = rollouts_batch.get('hallucination_image_mask', None)
                    if img_mask is not None:
                        img_mask = img_mask.to(device).float()
                        denom = img_mask.sum().clamp(min=1.0)
                        loss_cot_dpo_img = (cot_dpo_losses_img * img_mask).sum() / denom
                    else:
                        loss_cot_dpo_img = cot_dpo_losses_img.mean()

            # L_CoT_Anchor (RE Anchor)
            cot_anchor_losses = -F.logsigmoid(self.beta * (chosen_cot_logratios - self.anchor_value))
            loss_cot_anchor = cot_anchor_losses.mean()

            stats['debug/cot_dpo_base'] = loss_cot_dpo_base.detach()
            stats['debug/cot_anchor'] = loss_cot_anchor.detach()
            stats['cot/policy_positive_sum'] = policy_positive_cot_sum.mean().detach()
            stats['cot/ref_positive_sum'] = ref_positive_cot_sum.mean().detach()
            stats['cot/chosen_logratios'] = chosen_cot_logratios.mean().detach()

        # L_RE = Base + Instr_CoT + Img_CoT
        loss_RE = loss_cot_dpo_base + loss_cot_dpo_instr + loss_cot_dpo_img
        
        # L_DPO = L_gen + Instr_Full + Img_Full
        loss_DPO = loss_gen + loss_lif_instruction + loss_lif_image
        
        # L_Anc = L_Anc_RE + lambda_DPO * L_Anc_DPO
        loss_Anc = loss_cot_anchor + self.lambda_dpo * loss_anchor
        
        # Total = L_RE + lambda_DPO * L_DPO + lambda_Anc * L_Anc
        total_loss = loss_RE + self.lambda_dpo * loss_DPO + self.lambda_anc * loss_Anc

        stats['loss/total'] = total_loss.detach()
        stats['loss/RE'] = loss_RE.detach()
        stats['loss/DPO'] = (self.lambda_dpo * loss_DPO).detach()
        stats['loss/Anc'] = (self.lambda_anc * loss_Anc).detach()
        
        stats['debug/gen'] = loss_gen.detach()
        stats['debug/lif_instruction'] = loss_lif_instruction.detach()
        stats['debug/lif_image'] = loss_lif_image.detach()
        stats['debug/anchor'] = loss_anchor.detach()
        stats['logprobs/policy_positive'] = policy_positive_total.mean().detach()
        stats['logprobs/policy_negative'] = policy_negative_total.mean().detach()
        stats['logprobs/ref_positive'] = ref_positive_total.mean().detach()
        stats['logprobs/ref_negative'] = ref_negative_logprobs.mean().detach()
        stats['logratios/chosen_gen'] = chosen_logratios_gen.mean().detach()
        stats['logratios/rejected_gen'] = rejected_logratios_gen.mean().detach()
        stats['rewards/margin_gen'] = (self.beta * logits_gen).mean().detach()
        return total_loss, stats

    def record_step_stats(self, rollouts, train_stats, step_idx):
        stats = {}

        if self.optimizer is not None:
            stats['objective/lr'] = self.optimizer.param_groups[0]['lr']

        for k, v in train_stats.items():
            stats[f"dpo/{k}"] = v.mean(dim=0)

        stats = {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in stats.items()
        }

        return stats

    def save_model(self, output_dir):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(f"\nSaving model checkpoint to {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            unwrapped_policy = self.accelerator.unwrap_model(self.policy)
            unwrapped_policy.model.set_adapter("lora_policy")
            unwrapped_policy.model.save_pretrained(
                output_dir,
                selected_adapters=["lora_policy"],
            )

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            if self.optimizer is not None:
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(output_dir, OPTIMIZER_NAME)
                )

            if self.lr_scheduler is not None:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME)
                )

            print("Checkpoint saved successfully!")
        self.accelerator.wait_for_everyone()

    def resume_training(self, checkpoint_dir):
        optimizer_path = os.path.join(checkpoint_dir, OPTIMIZER_NAME)
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu")
            )

        scheduler_path = os.path.join(checkpoint_dir, SCHEDULER_NAME)
        if os.path.exists(scheduler_path):
            self.lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu")
            )

        m_step = re.search(r"checkpoint-step-(\d+)", checkpoint_dir)
        m_epoch = re.search(r"checkpoint-epoch-(\d+)", checkpoint_dir)

        if m_step:
            skipping_steps = int(m_step.group(1))
        elif m_epoch:
            steps_per_epoch = len(self.train_dataset) // self.args.rollout_batch_size
            steps_per_epoch = max(1, steps_per_epoch) 
            skipping_steps = int(m_epoch.group(1)) * steps_per_epoch
        else:
            skipping_steps = 0

        return skipping_steps
