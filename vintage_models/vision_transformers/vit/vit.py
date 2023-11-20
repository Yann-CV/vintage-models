from torch.nn import Module, LayerNorm, Linear, Parameter
from torch import Tensor, randn, cat, softmax

from vintage_models.components.attention import MultiHeadAttention
from vintage_models.components.multilayer_perceptron import TwoLayerGeluMLP, TwoLayerMLP
from vintage_models.components.patch import PatchConverter
from vintage_models.components.positional_encoding import LearnablePositionalEncoding1D
from vintage_models.components.residual import ResidualWithSelfAttention
from vintage_models.utility.image import PaddingMode


class ViTEncoder(Module):
    def __init__(
        self,
        layer_count: int,
        head_count: int,
        embedding_len: int,
        mlp_hidden_size: int,
    ) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.self_attention_residual = ResidualWithSelfAttention(
            [
                LayerNorm(embedding_len),
                MultiHeadAttention(dk=embedding_len, dv=embedding_len, h=head_count),
            ]
        )
        self.mlp_residual = ResidualWithSelfAttention(
            [
                LayerNorm(embedding_len),
                TwoLayerGeluMLP(
                    in_size=embedding_len,
                    out_size=embedding_len,
                    hidden_size=mlp_hidden_size,
                ),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.layer_count):
            x = self.mlp_residual(self.self_attention_residual(x))
        return x


class ViTEmbedder(Module):
    def __init__(
        self,
        patch_size: int,
        image_width: int,
        image_height: int,
        embedding_len: int,
    ) -> None:
        super().__init__()

        self.linear = Linear(
            in_features=patch_size * patch_size * 3,
            out_features=embedding_len,
        )

        self.positional_encoding = LearnablePositionalEncoding1D(
            sequence_len=(image_width // patch_size) * (image_height // patch_size) + 1,
            embedding_len=embedding_len,
        )
        self.cls_token = Parameter(randn(1, embedding_len))

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = cat((self.cls_token, x), dim=0)
        return self.positional_encoding(x)


class ViT(Module):
    def __init__(
        self,
        patch_size: int,
        image_width: int,
        image_height: int,
        embedding_len: int,
        mlp_hidden_size: int,
        head_count: int,
        layer_count: int,
        class_count: int,
        padding_mode: PaddingMode = PaddingMode.CONSTANT,
    ) -> None:
        super().__init__()
        self.patch_converter = PatchConverter(
            patch_size=patch_size,
            image_width=image_width,
            image_height=image_height,
            padding_mode=padding_mode,
        )
        self.embedder = ViTEmbedder(
            patch_size=patch_size,
            image_width=image_width,
            image_height=image_height,
            embedding_len=embedding_len,
        )
        self.encoder = ViTEncoder(
            layer_count=layer_count,
            head_count=head_count,
            embedding_len=embedding_len,
            mlp_hidden_size=mlp_hidden_size,
        )
        self.classification_mlp = TwoLayerMLP(
            in_size=embedding_len,
            hidden_size=mlp_hidden_size,
            out_size=class_count,
        )

    def forward(self, x: Tensor) -> Tensor:
        image_as_patch = self.patch_converter(x)
        patch_embeddings = self.embedder(image_as_patch)
        encoded = self.encoder(patch_embeddings)
        classified = self.classification_mlp(encoded[0, :])
        return softmax(classified, dim=-1)
