import torch
from torch import nn
from torch.nn.functional import gelu

from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig, MulticellularTransformerOutput


def _load_checkpoint_config(pretrained_ckpt_path: str) -> MulticellularGraphConfig:
    checkpoint = torch.load(pretrained_ckpt_path, map_location="cpu")
    return checkpoint["hyper_parameters"]["config"]


class MulticellularTransformerEncoder(nn.Module):
    """MulticellularTransformer encoder."""

    def __init__(self, config: MulticellularGraphConfig, pretrained_ckpt_path: str = None):
        super().__init__()
        self.config = config

        self.mask_token_idx = config.mask_token_idx
        self.padding_token_idx = config.padding_token_idx
        self.pretrained_ckpt_path = pretrained_ckpt_path

        pretrained_config = config
        if pretrained_ckpt_path is not None:
            pretrained_config = _load_checkpoint_config(pretrained_ckpt_path)
            config.gene_emb_dim = pretrained_config.gene_emb_dim

        self.gene_id_emb = nn.Embedding(
            num_embeddings=pretrained_config.n_genes,
            embedding_dim=pretrained_config.gene_emb_dim,
            padding_idx=pretrained_config.padding_token_idx,
        )
        if config.n_tissues > 0:
            self.tissue_emb = nn.Embedding(
                num_embeddings=config.n_tissues,
                embedding_dim=pretrained_config.gene_emb_dim,
                padding_idx=config.padding_token_idx,
            )

        # 2-layer MLP for gene expression
        self.gene_expression_encoder = nn.Sequential(
            nn.Linear(
                in_features=config.n_input,
                out_features=config.hidden_dim,
            ),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                in_features=config.hidden_dim,
                out_features=config.gene_emb_dim,
            ),
        )
        # creating a mask embedding for gene expression
        self.gene_expression_mask_emb = nn.Embedding(num_embeddings=1, embedding_dim=config.gene_emb_dim)

        self.layer_norm = nn.LayerNorm(config.gene_emb_dim)
        self.dropout = nn.Dropout(config.dropout)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pretrained_config.gene_emb_dim,
            nhead=pretrained_config.n_heads,
            dim_feedforward=pretrained_config.gene_emb_dim * 4,
            dropout=pretrained_config.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=pretrained_config.n_layers)

    def forward(
        self,
        expression_matrix: torch.Tensor,
        gene_ids: torch.Tensor,
        is_masked_gene_expression: torch.Tensor,
        padding_mask: torch.Tensor = None,
        tissue: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        # get gene embeddings
        # get gene id embeddings
        gene_id_emb = self.gene_id_emb(gene_ids)

        # get gene expression embeddings
        gene_expression_emb = self.gene_expression_encoder(expression_matrix)

        if is_masked_gene_expression.sum() > 0:
            gene_expression_emb[is_masked_gene_expression == 1] = self.gene_expression_mask_emb(
                torch.tensor(0, device=gene_expression_emb.device)
            ).type_as(gene_expression_emb)

        # sum gene id and gene expression embeddings
        gene_emb = gene_id_emb + gene_expression_emb

        if tissue is not None:
            tissue_emb = self.tissue_emb(tissue)[:, None, :]
            # add tissue embedding as a token
            gene_emb = torch.concatenate([tissue_emb, gene_emb], dim=1)
            padding_mask = torch.cat([torch.zeros_like(tissue).bool().unsqueeze(-1), padding_mask], dim=1)

        gene_emb = self.layer_norm(gene_emb)
        gene_emb = self.dropout(gene_emb)

        encoder_output = self.encoder(gene_emb, src_key_padding_mask=padding_mask)

        return encoder_output


class MulticellularTransformerModule(nn.Module):
    """MulticellularTransformer module."""

    def __init__(self, config: MulticellularGraphConfig, pretrained_ckpt_path: str = None):
        super().__init__()

        self.encoder = MulticellularTransformerEncoder(config, pretrained_ckpt_path)

        self.dropout = nn.Dropout(config.dropout)

        pretrained_config = config
        if pretrained_ckpt_path is not None:
            pretrained_config = _load_checkpoint_config(pretrained_ckpt_path)

        self.gene_id_classifier = MulticellularTransformerMGMHead(pretrained_config)
        self.tissue_classifier = nn.Linear(pretrained_config.gene_emb_dim, config.n_tissues)
        self.gex_decoder = MulticellularTransformerDecoderHead(config)

    def forward(
        self,
        expression_matrix: torch.Tensor,
        gene_ids: torch.Tensor,
        is_masked_gene_expression: torch.Tensor,
        padding_mask: torch.Tensor = None,
        tissue: torch.Tensor = None,
    ) -> MulticellularTransformerOutput:
        """Forward pass of the model."""
        last_hidden_state = self.encoder(
            expression_matrix,
            gene_ids,
            is_masked_gene_expression,
            padding_mask,
            tissue,
        )
        # remove the tissue embedding from the last hidden state as we dont need it anymore
        tissue_logits = None
        if tissue is not None:
            tissue_hidden_state = last_hidden_state[:, 0, :]
            last_hidden_state = last_hidden_state[:, 1:, :]
            tissue_logits = self.tissue_classifier(self.dropout(tissue_hidden_state))

        # get gene id predictions
        gene_id_logits = self.gene_id_classifier(self.dropout(last_hidden_state))

        # get gene expression reconstruction
        gex_hat = self.gex_decoder(self.dropout(last_hidden_state))

        return MulticellularTransformerOutput(
            last_hidden_state=last_hidden_state,
            gene_id_logits=gene_id_logits,
            tissue_logits=tissue_logits,
            gex_reconstruction=gex_hat,
        )


class MulticellularTransformerMGMHead(nn.Module):
    """MulticellularTransformer Masked Gene Modeling Head."""

    def __init__(self, config: MulticellularGraphConfig):
        super().__init__()
        self.dense = nn.Linear(config.gene_emb_dim, config.gene_emb_dim)
        self.layer_norm = nn.LayerNorm(config.gene_emb_dim, eps=config.layer_norm_eps)

        # add 1 to the condfig.protein_clusters_vocab_size to account for the end token
        self.decoder = nn.Linear(config.gene_emb_dim, config.n_genes, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.n_genes))

    def forward(self, features, **kwargs):
        """Forward method for the head."""
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to nr of labels with bias
        x = self.decoder(x) + self.bias
        return x


class MulticellularTransformerDecoderHead(nn.Module):
    """MulticellularTransformer Masked Gene Modeling Head."""

    def __init__(self, config: MulticellularGraphConfig):
        super().__init__()
        self.dense = nn.Linear(config.gene_emb_dim, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_dim, config.n_input, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.n_input))
        self.softplus = nn.Softplus()

    def forward(self, features, **kwargs):
        """Forward method for the head."""
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to nr of labels with bias
        x = self.decoder(x) + self.bias
        return self.softplus(x)
