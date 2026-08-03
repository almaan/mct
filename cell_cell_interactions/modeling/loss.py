import torch
from torch.distributions import Normal
from torch.nn.functional import mse_loss


def compute_masked_gene_id_loss(
    gene_id_logits: torch.Tensor, labels: torch.Tensor, is_masked_gene_id: torch.Tensor
) -> torch.Tensor:
    """Compute the loss for the masked gene ids."""
    labels = labels[is_masked_gene_id == 1]
    gene_id_logits = gene_id_logits[is_masked_gene_id == 1]

    if labels.size(0) == 0:
        return torch.tensor(0.0, device=gene_id_logits.device)

    # compute the loss
    loss = torch.nn.functional.cross_entropy(gene_id_logits, labels)
    return loss


def compute_elbo(
    z_posterior: torch.distributions.Normal,
    gex_reconstruction: torch.Tensor,
    expression_matrix: torch.Tensor,
):
    """Compute ELBO."""
    rec_loss = mse_loss(gex_reconstruction, expression_matrix, reduction="none").sum(-1).mean()

    z_prior = Normal(torch.zeros_like(z_posterior.mean), torch.ones_like(z_posterior.stddev))
    kl_div = torch.distributions.kl.kl_divergence(z_posterior, z_prior).sum(-1).mean()
    return rec_loss, kl_div
