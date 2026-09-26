import torch


class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def reset(self):
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def merge_dict(dicts, merge_fn=lambda *args: args):
    if len(dicts) == 0:
        return dict()
    return {key: merge_fn([dict_[key] for dict_ in dicts]) for key in dicts[0].keys()}


def unpack_dict(d, keys, return_type=tuple):
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def flatten_dict(nested, sep=".", postprocess_fn=lambda *args: args):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):
                rec(v, prefix + k + sep, into)
            else:
                v = postprocess_fn(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def masked_mean(values, mask, axis=None):
    if axis is not None:
        return (values * mask).sum(axis=axis, keepdim=True) / mask.sum(axis=axis, keepdim=True)
    else:
        return (values * mask).sum() / mask.sum()


def prepare_inputs(data, device):
    if isinstance(data, dict):
        return type(data)({k: prepare_inputs(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_inputs(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def create_optimizer(args, model, optimizer=None):
    if optimizer is not None:
        return optimizer

    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=getattr(args, 'weight_decay', 0.0),
        betas=(getattr(args, 'adam_beta1', 0.9), getattr(args, 'adam_beta2', 0.999)),
        eps=getattr(args, 'adam_epsilon', 1e-8),
    )


def create_scheduler(
    args,
    optimizer,
    lr_scheduler=None,
    num_training_steps=None,
    num_warmup_steps=None,
):
    if lr_scheduler is not None:
        return lr_scheduler

    lr_scheduler_type = getattr(args, 'lr_scheduler_type', None)

    if lr_scheduler_type is None:
        return None

    if num_warmup_steps is None:
        num_warmup_steps = getattr(args, 'warmup_steps', 0)

    if lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif lr_scheduler_type == "constant":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    else:
        raise ValueError(f"Unknown lr_scheduler_type: {lr_scheduler_type}")


def compute_grad_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]),
        2.0,
    )
    return total_norm
