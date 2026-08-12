"""Training invariants for Task-Conditioned Mixture Routing (TCMR)."""

import torch


def task_conditioned_moe_batch_step(model, batch, optimizer, criterion,
                                    consistency_weight=0.01):
    """Optimize one original batch with exactly one optimizer step.

    Binary cross entropy and the optional routing consistency term are both
    sample-proportion weighted across task sub-batches. This preserves the
    full-batch sample objective while allowing task-isolated forward/backward
    passes. The function deliberately contains the sole ``optimizer.step`` used
    for the original batch so the invariant can be tested without loading data
    or metric dependencies.
    """
    task_batch = batch["tab"]
    optimizer.zero_grad()
    base_loss_total = 0.0
    auxiliary_loss_total = 0.0

    for task_id in task_batch.unique():
        mask = task_batch == task_id
        sub_batch = {key: value[mask] for key, value in batch.items()}
        model(sub_batch)
        sample_weight = mask.sum().to(torch.float32) / task_batch.numel()
        base_loss = criterion(
            sub_batch["logit"], sub_batch["is_click"].float(),
        ) * sample_weight
        auxiliary_loss = sub_batch.get(
            "_routing_auxiliary_loss", base_loss.new_zeros(()),
        ) * sample_weight
        loss = base_loss + consistency_weight * auxiliary_loss
        loss.backward()
        base_loss_total += float(base_loss.detach())
        auxiliary_loss_total += float(auxiliary_loss.detach())

    optimizer.step()
    return {
        "base_loss": base_loss_total,
        "routing_auxiliary_loss": auxiliary_loss_total,
        "optimizer_steps": 1,
    }
