from typing import Sequence
import numpy as np
import torch
from torch import nn
from fuse.dl.models.backbones.backbone_transformer import Transformer


class ProjectPatchesTokenizer(nn.Module):
    """
    Projects a 1D/2D/3D images to tokens using patches
    Assumes to have one of the forms:
    (1) batch_size, channels, height
    (2) batch_size, channels, height, width
    (3) batch_size, channels, height, width, depth

    The output shape is always:
    batch_size, num_tokens, token_dim
    """

    def __init__(self, *, image_shape: Sequence[int], patch_shape: Sequence[int], channels: int, token_dim: int):
        super().__init__()
        assert len(image_shape) == len(patch_shape), "patch and image must have identical dimensions"
        image_shape = np.array(image_shape)
        patch_shape = np.array(patch_shape)
        assert (image_shape % patch_shape == 0).all(), "Image dimensions must be divisible by the patch size."
        self.num_tokens = int(np.prod(image_shape // patch_shape))
        patch_shape = tuple(patch_shape)
        self.image_dim = len(image_shape)
        if self.image_dim == 1:
            self.proj_layer = nn.Conv1d(
                in_channels=channels, out_channels=token_dim, kernel_size=patch_shape, stride=patch_shape
            )
        elif self.image_dim == 2:
            self.proj_layer = nn.Conv2d(
                in_channels=channels, out_channels=token_dim, kernel_size=patch_shape, stride=patch_shape
            )
        elif self.image_dim == 3:
            self.proj_layer = nn.Conv3d(
                in_channels=channels, out_channels=token_dim, kernel_size=patch_shape, stride=patch_shape
            )
        else:
            raise NotImplementedError("only supports 1D/2D/3D images")

    def forward(self, x):
        assert len(x.shape) == self.image_dim + 2, "input should be [batch, channels] + image_shape"
        x = self.proj_layer(x)
        x = x.flatten(start_dim=2, end_dim=-1)  # x.shape == (batch_size, token_dim, num_tokens)
        x = x.transpose(1, 2)  # x.shape == (batch_size, num_tokens, token_dim)
        return x


class ViT(nn.Module):
    """
    Projects a 1D/2D/3D image into tokens, and then runs it through a transformer
    """

    def __init__(self, token_dim: int, projection_kwargs: dict, transformer_kwargs: dict):
        """
        :param token_dim: the dimension of each token in the transformer
        :param projection_kwargs: positional arguments for the ProjectPatchesTokenizer class
        :param transformer_kwargs: positional arguments for the Transformer class
        """
        super().__init__()
        self.projection_layer = ProjectPatchesTokenizer(token_dim=token_dim, **projection_kwargs)
        num_tokens = self.projection_layer.num_tokens
        self.transformer = Transformer(num_tokens=num_tokens, token_dim=token_dim, **transformer_kwargs)

    def forward(self, x: torch.Tensor, pool: str = "none"):
        """
        :param pool: returns all tokens (pool='none'), only cls token (pool='cls') or the average token (pool='mean')
        """
        assert pool in ["none", "cls", "mean"]
        x = self.projection_layer(x)
        x = self.transformer(x)
        if pool == "cls":
            x = x[:, 0]
        if pool == "mean":
            x = x.mean(dim=1)
        return x


def vit_tiny(
    image_shape: Sequence[int] = (224, 224), patch_shape: Sequence[int] = (16, 16), channels: int = 3
) -> nn.Module:
    token_dim = 192
    projection_kwargs = dict(image_shape=image_shape, patch_shape=patch_shape, channels=channels)
    transformer_kwargs = dict(depth=12, heads=3, mlp_dim=token_dim * 4, dim_head=64, dropout=0.0, emb_dropout=0.0)
    return ViT(token_dim=token_dim, projection_kwargs=projection_kwargs, transformer_kwargs=transformer_kwargs)


def vit_small(
    image_shape: Sequence[int] = (224, 224), patch_shape: Sequence[int] = (16, 16), channels: int = 3
) -> nn.Module:
    token_dim = 384
    projection_kwargs = dict(image_shape=image_shape, patch_shape=patch_shape, channels=channels)
    transformer_kwargs = dict(depth=12, heads=6, mlp_dim=token_dim * 4, dim_head=64, dropout=0.0, emb_dropout=0.0)
    return ViT(token_dim=token_dim, projection_kwargs=projection_kwargs, transformer_kwargs=transformer_kwargs)


def vit_base(
    image_shape: Sequence[int] = (224, 224), patch_shape: Sequence[int] = (16, 16), channels: int = 3
) -> nn.Module:
    token_dim = 768
    projection_kwargs = dict(image_shape=image_shape, patch_shape=patch_shape, channels=channels)
    transformer_kwargs = dict(depth=12, heads=12, mlp_dim=token_dim * 4, dim_head=64, dropout=0.0, emb_dropout=0.0)
    return ViT(token_dim=token_dim, projection_kwargs=projection_kwargs, transformer_kwargs=transformer_kwargs)


def usage_example():
    vit = vit_tiny()
    # an example input to the model
    x = torch.zeros([1, 3, 224, 224])
    print(f"image is projected into {vit.projection_layer.num_tokens} tokens")
    pred = vit(x, pool="cls")
    print(f"output shape is: {pred.shape}")
    return pred


if __name__ == "__main__":
    usage_example()
