import json
import os
from typing import Any

import torch
from torchmetrics.functional import accuracy, f1_score


def compute_classification_metrics(
    gene_id_logits: torch.Tensor, gene_ids: torch.Tensor, is_masked_gene_id: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute the classification metrics for the masked gene ids."""
    num_classes = gene_id_logits.size(-1)
    # exclude the non-masked tokens
    gene_id_logits = gene_id_logits[is_masked_gene_id == 1].cpu()
    gene_ids = gene_ids[is_masked_gene_id == 1].cpu()

    if gene_id_logits.size(0) == 0:
        dummy_value = torch.tensor(-100.0).type_as(gene_id_logits)
        return {"micro_acc": dummy_value, "macro_acc": dummy_value, "macro_f1": dummy_value}

    # compute the metrics
    micro_acc = accuracy(gene_id_logits, gene_ids, num_classes=num_classes, average="micro", task="multiclass")
    macro_acc = accuracy(gene_id_logits, gene_ids, num_classes=num_classes, average="macro", task="multiclass")
    macro_f1 = f1_score(gene_id_logits, gene_ids, num_classes=num_classes, average="macro", task="multiclass")

    return {"micro_acc": micro_acc, "macro_acc": macro_acc, "macro_f1": macro_f1}


def save_dicts(
    output_dir: str,
    dicts: list[dict[str, Any]],
    filenames: list[str],
):
    """Save the dictionaries to the output directory."""
    for filename, d in zip(filenames, dicts):
        with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
            json.dump(d, f)
