import os
import gc
import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import (
    InfiniteLoader,
    merge_dict,
    create_optimizer,
    create_scheduler,
    compute_grad_norm,
)


class RLTrainer(ABC):

    def __init__(
        self,
        args,
        train_dataset,
        eval_dataset,
        data_collator,
        policy,
        ref_policy,
        accelerator,
        optimizer=None,
        lr_scheduler=None,
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.policy = policy
        self.ref_policy = ref_policy
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
    def train(self, resume_training_ckpt= None):
        total_steps = self._compute_total_steps()
        total_optimizer_steps = self._compute_total_optimizer_steps(total_steps)
        steps_per_epoch = len(self.train_dataset) // self.args.rollout_batch_size
        steps_per_epoch = max(1, steps_per_epoch)  

        if self.accelerator.is_main_process:
            print(f"\n{'='*60}")
            print("Training Configuration:")
            print(f"{'='*60}")
            print(f"Total samples: {len(self.train_dataset)}")
            print(f"Total epochs: {self.args.total_epochs}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Rollout batch size: {self.args.rollout_batch_size}")
            print(f"Rollout per-device batch size: {self.args.rollout_per_device_batch_size}")
            print(f"Rollout accumulation steps: {self.args.rollout_accumulation_steps}")
            print(f"Step batch size: {self.args.step_batch_size}")
            print(f"Step per-device batch size: {self.args.step_per_device_batch_size}")
            print(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
            print(f"Noptepochs: {self.args.noptepochs}")
            print(f"Total rollout steps: {total_steps}")
            print(f"Total optimizer steps: {total_optimizer_steps}")
            print(f"{'='*60}\n")

        self.create_optimizer_and_scheduler(total_optimizer_steps)

        skipping_steps = 0
        if resume_training_ckpt is not None:
            skipping_steps = self.resume_training(resume_training_ckpt)
            if self.accelerator.is_main_process:
                print(f"Resuming from checkpoint: {resume_training_ckpt}")
                print(f"Skipping first {skipping_steps} steps\n")

        infinite_dataloader = self.get_train_dataloader()

        for epoch in range(1, self.args.total_epochs + 1):
            if epoch > 1:
                infinite_dataloader.reset()
            epoch_start_step = (epoch - 1) * steps_per_epoch + 1
            epoch_end_step = epoch * steps_per_epoch

            if epoch_end_step <= skipping_steps:
                for _ in range(steps_per_epoch * self.args.rollout_accumulation_steps):
                    next(infinite_dataloader)
                continue

            pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch}/{self.args.total_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for step_in_epoch in range(1, steps_per_epoch + 1):
                step_idx = epoch_start_step + step_in_epoch - 1

                if step_idx <= skipping_steps:
                    for _ in range(self.args.rollout_accumulation_steps):
                        next(infinite_dataloader)
                    pbar.update(1)
                    continue

                stats = self.step(infinite_dataloader, step_idx)

                if self.accelerator.is_main_process:
                    postfix = {}
                    for key, value in stats.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        if 'loss' in key.lower():
                            postfix[key.replace('loss/', '').replace('dpo/', '')] = f"{value:.4f}"

                    if 'total' in postfix:
                        ordered_postfix = {'total': postfix.pop('total')}
                        ordered_postfix.update(postfix)
                        postfix = ordered_postfix

                    pbar.set_postfix(postfix)
                    pbar.update(1)

                if step_idx % self.args.logging_steps == 0 and self.accelerator.is_main_process:
                    loss_str = f"Step {step_idx}/{total_steps} | "
                    loss_metrics = []
                    for key, value in stats.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        if 'loss' in key.lower():
                            loss_metrics.append(f"{key}: {value:.4f}")
                        elif 'lr' in key.lower():
                            loss_metrics.append(f"{key}: {value:.2e}")
                    loss_str += " | ".join(loss_metrics)
                    pbar.write(loss_str)

                if self.args.save_steps > 0 and step_idx % self.args.save_steps == 0:
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-step-{step_idx}")
                    self.save_model(checkpoint_dir)
                    if self.accelerator.is_main_process:
                        pbar.write(f"\nCheckpoint saved to: {checkpoint_dir}\n")

            pbar.close()

            epoch_checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch}")
            self.save_model(epoch_checkpoint_dir)
            if self.accelerator.is_main_process:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch} completed!")
                print(f"Checkpoint saved to: {epoch_checkpoint_dir}")
                print(f"{'='*60}\n")

        final_dir = os.path.join(self.args.output_dir, "checkpoint-final")
        self.save_model(final_dir)
        if self.accelerator.is_main_process:
            print(f"\n{'='*60}")
            print("Training completed!")
            print(f"Final model saved to: {final_dir}")
            print(f"{'='*60}\n")
        self.accelerator.end_training()
    
    def step(self, train_dataloader, step_idx):
        """Execute single training step"""
        queries_batches = [
            next(train_dataloader)
            for _ in range(self.args.rollout_accumulation_steps)
        ]

        rollouts = self.rollout(queries_batches)
        torch.cuda.empty_cache()

        train_stats = self.step_with_rollouts(rollouts)

        stats = self.record_step_stats(
            rollouts=rollouts,
            train_stats=train_stats,
            step_idx=step_idx,
        )

        return stats

    def step_with_rollouts(self, rollouts):
        rollouts_dataloader = self.get_rollouts_dataloader(rollouts)
        stats_list = []

        for _ in range(self.args.noptepochs):
            for rollouts_batch in rollouts_dataloader:

                gc.collect()
                torch.cuda.empty_cache()

                with self.accelerator.accumulate(self.policy):
                    stats_for_this_step = {}

                    policy_loss, policy_stats = self.compute_policy_loss(rollouts_batch)
                    stats_for_this_step.update(policy_stats)
                    self.accelerator.backward(policy_loss)

                    if self.accelerator.sync_gradients:
                        if self.args.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(
                                self.policy.parameters(),
                                self.args.max_grad_norm,
                            )
                        stats_for_this_step['loss/grad_norm'] = compute_grad_norm(self.policy)
                        stats_list.append(stats_for_this_step)

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

        return merge_dict(stats_list, torch.stack)
    
    def get_train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.args.rollout_per_device_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)
        return InfiniteLoader(train_dataloader)

    def get_rollouts_dataloader(self, rollouts, shuffle=True, drop_last=True):

        pixel_value_keys = [k for k in rollouts.keys() if 'pixel_values' in k]
        image_grid_thw_keys = [k for k in rollouts.keys() if 'image_grid_thw' in k]

        for pv_key in pixel_value_keys:
            prefix = pv_key.replace('_pixel_values', '')
            thw_key = f"{prefix}_image_grid_thw"

            if pv_key in rollouts and thw_key in rollouts:
                pv_list = rollouts[pv_key] 
                thw_list = rollouts[thw_key]

                per_sample_pv = []
                for pv_batch, thw_batch in zip(pv_list, thw_list):
                    if pv_batch is None or thw_batch is None:
                        batch_size = thw_batch.shape[0] if thw_batch is not None else 1
                        per_sample_pv.extend([None] * batch_size)
                    else:
                        num_patches_per_sample = (thw_batch[:, 0] * thw_batch[:, 1] * thw_batch[:, 2]).tolist()

                        start_idx = 0
                        for num_patches in num_patches_per_sample:
                            num_patches = int(num_patches)
                            per_sample_pv.append(pv_batch[start_idx:start_idx + num_patches])
                            start_idx += num_patches

                rollouts[pv_key] = per_sample_pv

        for thw_key in image_grid_thw_keys:
            if thw_key in rollouts:
                thw_list = rollouts[thw_key]
                per_sample_thw = []
                for thw_batch in thw_list:
                    if thw_batch is None:
                        per_sample_thw.append(None)
                    else:
                        for i in range(thw_batch.shape[0]):
                            per_sample_thw.append(thw_batch[i:i+1])
                rollouts[thw_key] = per_sample_thw

        all_pixel_and_thw_keys = pixel_value_keys + image_grid_thw_keys
        regular_keys = tuple(k for k, v in rollouts.items() if torch.is_tensor(v) and k not in all_pixel_and_thw_keys)

        def collate_rollouts(instances):
            indices = [inst[0] for inst in instances]
            batch = {
                key: torch.stack([inst[idx + 1] for inst in instances])
                for idx, key in enumerate(regular_keys)
            }
            for key in all_pixel_and_thw_keys:
                if key in rollouts:
                    val_list = rollouts[key]
                    batch_vals = [val_list[i] for i in indices if val_list[i] is not None]
                    if len(batch_vals) > 0:
                        if 'pixel_values' in key:
                            batch[key] = torch.cat(batch_vals, dim=0)
                        else: 
                            batch[key] = torch.cat(batch_vals, dim=0)
                    else:
                        batch[key] = None

            return batch

        class IndexedTensorDataset(torch.utils.data.Dataset):
            def __init__(self, *tensors):
                self.tensors = tensors
                self.length = tensors[0].shape[0] if len(tensors) > 0 else 0

            def __getitem__(self, index):
                return (index,) + tuple(tensor[index] for tensor in self.tensors)

            def __len__(self):
                return self.length

        dataset = IndexedTensorDataset(*[rollouts[key] for key in regular_keys])
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.step_per_device_batch_size,
            collate_fn=collate_rollouts,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return dataloader

    def create_optimizer_and_scheduler(self, num_training_steps):
        optimizer = create_optimizer(
            args=self.args,
            model=self.policy,
            optimizer=self.optimizer,
        )
        lr_scheduler = create_scheduler(
            args=self.args,
            optimizer=optimizer,
            lr_scheduler=self.lr_scheduler,
            num_training_steps=num_training_steps * self.accelerator.num_processes,
            num_warmup_steps=self.args.warmup_steps * self.accelerator.num_processes,
        )

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            optimizer, lr_scheduler
        )

        if self.lr_scheduler is not None:
            self.accelerator.register_for_checkpointing(self.lr_scheduler)

    def _compute_total_steps(self):
        total_samples = len(self.train_dataset)
        total_epochs = self.args.total_epochs
        rollout_batch_size = self.args.rollout_batch_size
        steps_per_epoch = max(1, total_samples // rollout_batch_size)
        return steps_per_epoch * total_epochs

    def _compute_total_optimizer_steps(self, total_steps):
        optimizer_steps_per_rollout = self.args.rollout_batch_size // self.args.step_batch_size
        return total_steps * optimizer_steps_per_rollout * self.args.noptepochs

    @abstractmethod
    def rollout(self, queries_batches):
        raise NotImplementedError

    @abstractmethod
    def compute_policy_loss(self, rollouts_batch):
        raise NotImplementedError

    @abstractmethod
    def record_step_stats(self, rollouts, train_stats, step_idx):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, output_dir):
        raise NotImplementedError

    @abstractmethod
    def resume_training(self, checkpoint_dir):
        raise NotImplementedError
