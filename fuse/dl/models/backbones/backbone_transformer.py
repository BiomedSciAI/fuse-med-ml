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
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens + num_cls_tokens, token_dim)
        )
        self.cls_tokens = nn.Parameter(torch.randn(1, num_cls_tokens, token_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(
            dim=token_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_tokens, token_dim] shaped tensor
        :return: [batch_size, num_tokens + num_cls_tokens, token_dim] shaped tensor, where the first tokens are the CLS tokens
        """
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_tokens, "1 a d -> b a d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class CrossAttentionTransformerEncoder(nn.Module):
    """
    CrossAttentionTransformerEncoder backbone model based on x-transformers library.

    Input:
        two sequences 'seq_a, seq_b' with shapes [batch_size, len(seq_a)], [batch_size, len(seq_b)] respectively.
    Output:
        features tensor with shape [batch_size, output_dim]


    Architecture:
        The model consist of the following blocks:
            * Two encoding layers - one for each sequence
            * One or two cross attention (1) layers - depends on the user's request
            * One linear layer

    Forward Pass:
        -> Receive two sequences as inputs, assuming both of them are already tokenized.
        -> Pass each of the sequences through it's own encoder (extracting features)
        -> Performs cross attention with the two encoded features using one of the sequences as context
                (also supports using both as sequence:
                    * performs two times cross attention, each time using a different sequence as the context
                    * concat both outputs into one vector)
        -> Apply linear layer on the last encoded vector

    Building Blocks:
        In this architecture we use three components from "x-transformers" library.
        Here is a short summary - for more information consider check source code.
        * TransformerWrapper - wraps an attention layer (in our case the Encoder) and applies token & positional embedding.
        * Encoder - self attention layers. In our case it gets the embedding from the wrapper.
        * CrossAttender - cross attention layers. In our case it gets the embedding from the encoders.

    (1) see the following blog post for more info regarding cross attention in transformers:
        https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture
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
        kwargs_wrapper_a: Optional[dict] = None,
        kwargs_wrapper_b: Optional[dict] = None,
        kwargs_encoder_a: Optional[dict] = None,
        kwargs_encoder_b: Optional[dict] = None,
        kwargs_cross_attn: Optional[dict] = None,
    ):
        """
        :param emb_dim: inner model dimension
        :param num_tokens_a: number of tokens of the first sequence
        :param num_tokens_b: number of tokens of the second sequence
        :param max_seq_len_a: the maximum length of the first sequence
        :param max_seq_len_b: the maximum length of the second sequence
        :param depth_a: first sequence encoder's depth
        :param depth_b: second sequence encoder's depth
        :param depth_cross_attn: cross attender(s)' length
        :param heads_a: number of attention heads for the first sequence's encoder
        :param heads_b: number of attention heads for the second sequence's encoder
        :param output_dim: (optional) model's output dimension. if not give the emb dim will be used as default.
        :param context: which sequence will be used as context in the cross attention module:
                        "seq_a": the first sequence will be used as a context
                        "seq_b": the second sequence will be used as a context
                        "both": will use two cross attention modules to take each one of the sequences as a context to the other one.
        :param kwargs_wrapper_a: optional - additional arguments for sequence a's TransformerWrapper object
        :param kwargs_wrapper_b: optional - additional arguments for sequence b's TransformerWrapper object
        :param kwargs_encoder_a: optional - additional arguments for sequence a's Encoder object
        :param kwargs_encoder_b: optional - additional arguments for sequence b's Encoder object
        :param kwargs_cross_attn: optional - additional arguments for the CrossAttender object(s)
        """
        super().__init__()

        if output_dim is None:
            output_dim = emb_dim

        assert context in ["seq_a", "seq_b", "both"]
        self._context = context

        # init sequences' encoders
        self.enc_a = TransformerWrapper(
            num_tokens=num_tokens_a,
            max_seq_len=max_seq_len_a,
            **kwargs_wrapper_a if kwargs_wrapper_a else dict(),
            attn_layers=Encoder(
                dim=emb_dim,
                depth=depth_a,
                heads=heads_a,
                **kwargs_encoder_a if kwargs_encoder_a else dict(),
            ),
        )
        self.enc_b = TransformerWrapper(
            num_tokens=num_tokens_b,
            max_seq_len=max_seq_len_b,
            **kwargs_wrapper_b if kwargs_wrapper_b else dict(),
            attn_layers=Encoder(
                dim=emb_dim,
                depth=depth_b,
                heads=heads_b,
                **kwargs_encoder_b if kwargs_encoder_b else dict(),
            ),
        )

        # cross attention module(s)
        if self._context in ["seq_a", "seq_b"]:
            self.cross_attn = CrossAttender(
                dim=emb_dim,
                depth=depth_cross_attn,
                **kwargs_cross_attn if kwargs_cross_attn else dict(),
            )

        else:  # both
            self.cross_attn_a_as_context = CrossAttender(
                dim=emb_dim,
                depth=depth_cross_attn,
                **kwargs_cross_attn if kwargs_cross_attn else dict(),
            )
            self.cross_attn_b_as_context = CrossAttender(
                dim=emb_dim,
                depth=depth_cross_attn,
                **kwargs_cross_attn if kwargs_cross_attn else dict(),
            )

        self.last_linear = nn.Linear(emb_dim, output_dim)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """
        assumes input sequences are already tokenized

        :param xa: tensor with shape [batch_size, seq_len_a]
        :param xb: tensor with shape [batch_size, seq_len_b]
        :return: raw embeddings
        """
        # encoding stage
        enc_xa = self.enc_a(xa, return_embeddings=True)
        enc_xb = self.enc_b(xb, return_embeddings=True)

        # cross attention stage
        if self._context == "seq_a":
            x = self.cross_attn(enc_xb, context=enc_xa)

        elif self._context == "seq_b":
            x = self.cross_attn(enc_xa, context=enc_xb)

        else:
            x_acb = self.cross_attn_b_as_context(enc_xa, context=enc_xb)
            x_bca = self.cross_attn_a_as_context(enc_xb, context=enc_xa)
            x = torch.cat((x_acb, x_bca), dim=1)

        # linear layer
        x = self.last_linear(x)
        return x
