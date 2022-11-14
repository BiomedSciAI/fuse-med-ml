import torch
from torch import nn
from vit_pytorch.vit import Transformer as _Transformer
from vit_pytorch.vit import repeat


class Transformer(nn.Module):
    """
    Transformer backbone.
    Gets a [batch_size, num_tokens, token_dim] shaped tensor
    Returns a [batch_size, num_tokens + 1, token_dim] shaped tensor, where the first token is the CLS token
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
        emb_dropout: float = 0.0
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens + 1, token_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(
            dim=token_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_tokens, token_dim] shaped tensor
        :return: [batch_size, num_tokens + 1, token_dim] shaped tensor, where the first token is the CLS token
        """
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        return x
