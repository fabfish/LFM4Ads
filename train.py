import json
import os
from copy import deepcopy

import torch
from torcheval.metrics import BinaryAUROC
from tqdm import tqdm

import fields
from dataset import Dataset, Split
from task_conditioned_mixture_routing_protocol import task_conditioned_moe_batch_step

# Default 10 workers keeps D18/D19/D20 historical numbers reproducible.
# Lower this (e.g. LFM_NUM_WORKERS=4) only when packing multiple jobs per GPU.
# NOTE: pin_memory + non_blocking change only the *transport* of tensors from
# host to device; the compute graph and numeric order are unchanged, so AUC is
# bit-identical. Verified against pre-change runs on the same ckpt + seed.
NUM_WORKERS = int(os.environ.get("LFM_NUM_WORKERS", "10"))


def infer(model, dataset, train=False, shuffle=False, batch_size=10000,
          num_workers=None):
    model.train(train)
    device = next(model.parameters()).device
    workers = NUM_WORKERS if num_workers is None else num_workers
    loader = torch.utils.data.DataLoader(
        Dataset(dataset),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        shuffle=shuffle,
    )
    for batch in tqdm(loader):
        for field in fields.all:
            batch[field] = batch[field].to(device, non_blocking=True).int()
        with torch.inference_mode(not train):
            model(batch)
            yield batch


def evaluate(model, dataset, batch_size=10000, num_workers=None):
    AUC = BinaryAUROC()
    for batch in infer(
            model, dataset, batch_size=batch_size, num_workers=num_workers):
        AUC.update(batch["logit"], batch["is_click"].float())
    return AUC.compute()


def _trainable(model):
    """Only optimize parameters with requires_grad=True.

    Needed by the router+experts-only ablation (`--freeze dnn,head,sparse`) and
    by the frozen-backbone downstream heads; equivalent to `model.parameters()`
    when nothing is frozen.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters — check the --freeze setting.")
    return params


def train(model, scenario, shuffle=False):
    """Downstream training loop.

    `shuffle=False` is the historical default and keeps every D18/D19 result
    reproducible. Note that with shuffle=False the only stochastic source in a
    downstream trial is the head's random init, so `--seed` alone understates
    true run-to-run variance (same class of trap as defect G-4 upstream).
    """
    train_set, valid_set, test_set = Split(scenario)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(_trainable(model))
    auc_best = 0
    while True:
        for batch in infer(model, train_set, True, shuffle=shuffle):
            criterion(batch["logit"], batch["is_click"].float()).backward()
            optimizer.step()
            optimizer.zero_grad()
        auc = evaluate(model, valid_set)
        print(f"valid AUC: {auc:.4f}")
        if auc_best < auc - 0.001:
            auc_best = auc
            state_dict = deepcopy(model.state_dict())
        else:
            model.load_state_dict(state_dict)
            return evaluate(model, test_set)


# ============================================================
#  MoE training with gradient tracking
# ============================================================

def scenario_loss(criterion, logit, label, mask, tab_batch, loss_weighting):
    """Per-scenario BCE with optional sample-proportion weighting.

    - ``equal``: return the in-scenario mean (historical ``train_moe`` behaviour).
    - ``sample``: scale the in-scenario mean by ``|B_s| / |B|`` so that summing
      across scenarios equals the full-batch mean loss (see driver doc §二/§六).
      Only the *base* BCE is scaled; any specialization regularizer added later
      in ``train_moe`` is NOT scaled by this factor.

    Weight formula uses ``mask.sum()`` (nonzero count of a boolean mask), never
    ``len(sub)`` (dict field count) nor ``numel(mask)`` (always == numel(tab_batch)).
    """
    base = criterion(logit, label)
    if loss_weighting == "sample":
        w = mask.sum().to(base.dtype) / tab_batch.numel()
        return base * w
    if loss_weighting == "equal":
        return base
    raise ValueError(f"unknown loss_weighting={loss_weighting!r} "
                     f"(expected 'equal' or 'sample')")


def train_moe(model, scenario, tracker=None, spec_loss=None, lr=1e-3,
              beta2=0.999, shuffle=False, loss_weighting="equal"):
    """Train DCNv2MoE with per-scenario forward+backward for clean AU tracking.

    Each batch is split by scenario; each sub-batch gets its own forward+backward.
    This naturally separates per-scenario gradients without retain_graph.

    Args:
        model: DCNv2MoE instance
        scenario: "all" or a specific scenario id
        tracker: GradientTracker instance (optional, for AU accumulation)
        spec_loss: SpecializationLoss instance (optional, for encouraging specialization)
        loss_weighting: "equal" (historical, per-scenario mean, no scaling) or
            "sample" (scale each sub-batch BCE by its sample proportion so the
            summed objective equals the full-batch objective). Default "equal"
            preserves historical behaviour and all existing artifacts.
    Returns:
        test AUC (float)
    """
    # Refuse undefined combinations: sample weighting changes the *base* objective
    # only; the specialization regularizer's own weighting semantics are unset, so
    # silently mixing the two would change regularization strength. (driver doc §六)
    if loss_weighting == "sample" and spec_loss is not None and spec_loss.enabled:
        raise ValueError(
            "sample loss weighting with an enabled spec_loss is not defined: the "
            "specialization regularizer's sample-weighting semantics are unset. "
            "Refusing to run (see driver doc §六)."
        )

    train_set, valid_set, test_set = Split(scenario)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(_trainable(model), lr=lr,
                                  betas=(0.9, beta2))
    device = next(model.parameters()).device
    auc_best = 0
    epoch = 0
    ratios = {}
    all_gates = []  # collect gates for spec_loss EMA update
    all_tabs = []

    while True:
        epoch += 1
        loader = torch.utils.data.DataLoader(
            Dataset(train_set), batch_size=10000, num_workers=NUM_WORKERS,
            pin_memory=True, shuffle=shuffle,
        )
        all_gates.clear()
        all_tabs.clear()

        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            model.train()
            for field in fields.all:
                batch[field] = batch[field].to(device, non_blocking=True).int()
            tab_batch = batch["tab"]

            # Per-scenario: forward → hook records gradient → backward
            for s in tab_batch.unique():
                mask = tab_batch == s
                si = int(s.item())

                sub = {k: v[mask] for k, v in batch.items()}
                if tracker is not None:
                    tracker.set_scenario(si)

                model(sub)
                loss = scenario_loss(
                    criterion, sub["logit"], sub["is_click"].float(),
                    mask, tab_batch, loss_weighting,
                )

                # Phase 3: add specialization loss on THIS sub-batch's own
                # gate+tab (fix D5: previously only the leaked last sub was used).
                # sub["_gate"] is the LIVE gate (DCNv2MoE.forward stores the
                # non-detached gate), so spec.backward() reaches the router.
                if spec_loss is not None and spec_loss.enabled:
                    loss = loss + spec_loss.compute(
                        sub["_gate"], sub["tab"], ratios
                    )

                loss.backward()

                # Collect gates for EMA (every sub-batch, fixes D5)
                if spec_loss is not None and "_gate" in sub:
                    all_gates.append(sub["_gate"])  # list of 3 × [B_sub, K]
                    all_tabs.append(sub["tab"])

            optimizer.step()
            optimizer.zero_grad()

        # Epoch-end: update dominance ratios
        # Fix D1-B: key as 3-tuple (li, ei, s) so compute() lookup matches.
        if tracker is not None and spec_loss is not None:
            for li in range(3):
                dm = tracker.dominance_matrix(li)
                for (ei, s), v in dm.items():
                    ratios[(li, ei, s)] = v
            # Fix D1-A: check every epoch (MoE training lasts only 2-3 epochs).
            if not spec_loss.enabled:
                if spec_loss.check_and_enable(tracker):
                    print(f"[Epoch {epoch}] Specialization detected! Enabling spec loss.")

        # Epoch-end: update gate EMA from collected gates
        if spec_loss is not None and all_gates:
            # all_gates is list of per-sub-batch gate lists
            for sub_gates, sub_tab in zip(all_gates, all_tabs):
                spec_loss.update(sub_gates, sub_tab)

        auc = evaluate(model, valid_set)
        print(f"Epoch {epoch} valid AUC: {auc:.4f}")

        if tracker is not None and epoch % 10 == 0:
            print(tracker.summary())

        if auc_best < auc - 0.001:
            auc_best = auc
            state_dict = deepcopy(model.state_dict())
        else:
            model.load_state_dict(state_dict)
            return evaluate(model, test_set)


# ============================================================
#  Task-Conditioned Mixture Routing (TCMR) fair training
# ============================================================


def train_task_conditioned_moe(
        model, scenario="all", lr=1e-3, beta2=0.999, batch_size=10000,
        shuffle=True, max_epochs=300, consistency_weight=0.01,
        num_workers=None, epoch_probe=None, max_batches_per_epoch=None):
    """Train TCMR with sample weighting and one step per original batch.

    ``epoch_probe`` is an optional callback used for fixed validation-probe
    routing diagnostics. ``max_batches_per_epoch`` exists only for smoke tests;
    formal runs must leave it as ``None``.
    """
    train_set, valid_set, _ = Split(scenario)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        _trainable(model), lr=lr, betas=(0.9, beta2),
    )
    device = next(model.parameters()).device
    workers = NUM_WORKERS if num_workers is None else num_workers
    best_auc = float("-inf")
    best_state = None
    history = []

    for epoch in range(1, max_epochs + 1):
        loader = torch.utils.data.DataLoader(
            Dataset(train_set), batch_size=batch_size, num_workers=workers,
            pin_memory=True, shuffle=shuffle,
        )
        epoch_base_loss = 0.0
        epoch_auxiliary_loss = 0.0
        optimizer_steps = 0
        for batch_index, batch in enumerate(
                tqdm(loader, desc=f"TCMR Epoch {epoch}"), start=1):
            model.train()
            for field_name in fields.all:
                batch[field_name] = batch[field_name].to(
                    device, non_blocking=True,
                ).int()
            step_metrics = task_conditioned_moe_batch_step(
                model=model, batch=batch, optimizer=optimizer,
                criterion=criterion, consistency_weight=consistency_weight,
            )
            epoch_base_loss += step_metrics["base_loss"]
            epoch_auxiliary_loss += step_metrics["routing_auxiliary_loss"]
            optimizer_steps += step_metrics["optimizer_steps"]
            if (max_batches_per_epoch is not None
                    and batch_index >= max_batches_per_epoch):
                break

        valid_auc = float(evaluate(model, valid_set))
        record = {
            "epoch": epoch,
            "valid_auc": valid_auc,
            "base_loss_sum": epoch_base_loss,
            "routing_auxiliary_loss_sum": epoch_auxiliary_loss,
            "optimizer_steps": optimizer_steps,
        }
        if epoch_probe is not None:
            record["routing_probe"] = epoch_probe(model, epoch)
        history.append(record)
        print(
            f"TCMR Epoch {epoch} valid AUC={valid_auc:.4f} "
            f"optimizer_steps={optimizer_steps}"
        )

        if valid_auc > best_auc + 0.001:
            best_auc = valid_auc
            best_state = deepcopy(model.state_dict())
        else:
            break

    if best_state is None:
        raise RuntimeError("TCMR training produced no checkpointable epoch")
    model.load_state_dict(best_state)
    return history


# ============================================================
#  Continual learning training: sequential scenario training
# ============================================================

def train_continual(model, scenario, scenario_order=None):
    """Sequentially train on scenarios and evaluate on all after each task.

    Args:
        model: DCNv2 or DCNv2MoE
        scenario_order: list of scenario ids in training order.
                       If scenario == "moe", this is {"train": [...], "test": [...]}
                       If scenario == "base", just the order.
    Returns:
        list of per-task result dicts: [{task, scenario, auc_per_scenario}]
    """
    if scenario_order is None:
        scenario_order = [0, 1, 2, 3, 4, 5, 6, 8]

    criterion = torch.nn.BCEWithLogitsLoss()
    results = []

    for task_idx, train_scenario in enumerate(scenario_order):
        print(f"\n=== Task {task_idx}: training on scenario {train_scenario} ===")

        # Get train/valid/test for this scenario
        train_set, valid_set, test_set = Split(train_scenario)
        optimizer = torch.optim.AdamW(_trainable(model))
        auc_best = 0

        # Train on current scenario
        while True:
            for batch in infer(model, train_set, True):
                criterion(
                    batch["logit"], batch["is_click"].float()
                ).backward()
                optimizer.step()
                optimizer.zero_grad()
            auc = evaluate(model, valid_set)
            if auc_best < auc - 0.001:
                auc_best = auc
                state_dict = deepcopy(model.state_dict())
            else:
                model.load_state_dict(state_dict)
                break

        # Evaluate on ALL scenarios after training on this task
        auc_per_scenario = {}
        for eval_scenario in [0, 1, 2, 3, 4, 5, 6, 8]:
            _, _, test_set_s = Split(eval_scenario)
            auc_per_scenario[eval_scenario] = evaluate(
                model, test_set_s
            ).item()

        result = {
            "task": task_idx,
            "train_scenario": train_scenario,
            "auc_per_scenario": auc_per_scenario,
        }
        results.append(result)

        print(f"  AUC after task {task_idx}: "
              + " ".join(f"s{s}:{auc_per_scenario[s]:.4f}" for s in scenario_order))

    return results


def compute_forgetting(results, pre_continual=None):
    """Compute forgetting metrics from continual learning results.

    Forgetting on task j after training task i (i > j):
      F_{i,j} = AUC_after_task_i_training[s_j] - AUC_baseline[s_j]
    NEGATIVE value = forgetting occurred (AUC dropped relative to baseline).
    (The docstring previously claimed "Positive = forgetting occurred" — that was
    reversed; the implementation has always been `auc_current - auc_baseline`. See D1-C.)

    Baseline (D1-D fix, 2026-08-09): by default the PRE-CONTINUAL (pre-trained) AUC is
    used, so forgetting is measured relative to the model BEFORE any sequential training.
    Previously `results[0]` (AUC after task 0) was used, a less meaningful baseline.
    Pass `pre_continual=None` to fall back to the old `results[0]` behavior.
    """
    n_tasks = len(results)
    forgetting = []
    # D1-D fix: baseline = pre-trained AUC, not AUC after task 0.
    auc_baseline = (pre_continual if pre_continual is not None
                    else results[0]["auc_per_scenario"])

    for i in range(1, n_tasks):
        auc_current = results[i]["auc_per_scenario"]
        for j in range(i):
            forget = auc_current[results[j]["train_scenario"]] - auc_baseline[results[j]["train_scenario"]]
            forgetting.append({
                "train_task": i,
                "eval_task": j,
                "after_training_scenario": results[i]["train_scenario"],
                "eval_scenario": results[j]["train_scenario"],
                "forgetting": forget,
            })
    return forgetting
