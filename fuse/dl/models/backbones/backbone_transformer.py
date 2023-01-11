import torch
from torch import nn
from typing import Optional
from vit_pytorch.vit import Transformer as _Transformer
from vit_pytorch.vit import repeat
from x_transformers import Encoder, CrossAttender, TransformerWrapper


class Transformer(nn.Module):
    """
    Transformer backbone.
    Gets a [batch_size, num_tokens, token_dim] shaped tensor
    Returns a [batch_size, num_tokens + num_cls_tokens, token_dim] shaped tensor, where the first tokens are the CLS tokens
    """

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        num_cls_tokens: int = 1
    ):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens + num_cls_tokens, token_dim))
        self.cls_token = nn.Parameter(torch.randn(1, num_cls_tokens, token_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(
            dim=token_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_tokens, token_dim] shaped tensor
        :return: [batch_size, num_tokens + num_cls_tokens, token_dim] shaped tensor, where the first tokens are the CLS tokens
        """
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 a d -> b a d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class CrossAttentionTransformer(nn.Module):
    """

    "a" := first modality
    "b" := second modality

    """

    def __init__(
        self,
        emb_dim: int,
        num_tokens_a: int,
        num_tokens_b: int,
        max_seq_len_a: int,
        max_seq_len_b: int,
        depth_a: int = 6,
        depth_b: int = 6,
        depth_cross_attn: int = 6,
        heads_a: int = 9,
        heads_b: int = 9,
        output_dim: Optional[int] = None,
        context: str = "seq_b",
    ):
        """
        TODO:
        [ ] receive params as tuples?
        [x] support output_dim parameter
        [x] remove head (return features with size 'output_dim')
        [ ] clean and document
        [ ] supports two different emb_dim: for each sequence

        :param emb_dim: model dimension
        :param context:

        drug -> a
        target -> b
        """
        super().__init__()

        if output_dim is None:
            output_dim = emb_dim

        assert context in ["seq_a", "seq_b", "both"]
        self._context = context

        self.enc_a = TransformerWrapper(
            num_tokens=num_tokens_a,
            max_seq_len=max_seq_len_a,
            attn_layers=Encoder(dim=emb_dim, depth=depth_a, heads=heads_a),
        )
        self.enc_b = TransformerWrapper(
            num_tokens=num_tokens_b,
            max_seq_len=max_seq_len_b,
            attn_layers=Encoder(dim=emb_dim, depth=depth_b, heads=heads_b),
        )

        if self._context in ["seq_a", "seq_b"]:
            self.cross_attn = CrossAttender(dim=emb_dim, depth=depth_cross_attn)

        else:  # both
            self.cross_attn_a_as_context = CrossAttender(dim=emb_dim, depth=depth_cross_attn)
            self.cross_attn_b_as_context = CrossAttender(dim=emb_dim, depth=depth_cross_attn)
            self.ff = nn.Linear(emb_dim * 2, emb_dim)

        self.last_linear = nn.Linear(emb_dim, output_dim)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        enc_xa = self.enc_a(xa, return_embeddings=True)
        enc_xb = self.enc_b(xb, return_embeddings=True)

        if self._context == "seq_a":
            x = self.cross_attn(enc_xb, context=enc_xa)
            x = x[:, 0]

        if self._context == "seq_b":
            x = self.cross_attn(enc_xa, context=enc_xb)
            x = x[:, 0]

        if self._context == "both":
            x_acb = self.cross_attn_b_as_context(enc_xa, context=enc_xb)
            x_bca = self.cross_attn_a_as_context(enc_xb, context=enc_xa)
            x = torch.cat((x_acb[:, 0], x_bca[:, 0]), dim=1)
            x = self.ff(x)

        x = self.last_linear(x)
        # ---
        # request output dim
        # two diff emb dim

        # TODO delete
        # x = self.head(x)
        # logits = x

        # cls_preds = F.softmax(logits, dim=1)
        # return logits, cls_preds

        return x
