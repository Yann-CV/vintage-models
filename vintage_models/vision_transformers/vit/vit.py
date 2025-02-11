from torch.nn import Module, LayerNorm, Linear, Parameter, Sequential
from torch import Tensor, randn, cat, softmax

from vintage_models.components.attention import MultiHeadAttention
from vintage_models.components.multilayer_perceptron import TwoLayerGeluMLP, TwoLayerMLP
from vintage_models.components.patch import PatchConverter
from vintage_models.components.positional_encoding import LearnablePositionalEncoding1D
from vintage_models.components.residual import ResidualWithSelfAttention
from vintage_models.utility.transform import PaddingMode

class ViTEncoderLayer(Module):
    """Layer of the ViT encoder

    Attributes:
        head_count: The number of attention heads to use in each layer.
        embedding_len: The length of the embedding.
        mlp_hidden_size: The size of the hidden layer of the MLP.
        self_attention_residual: The residual module for the self attention (attention plus normalization).
        mlp_residual: The residual module for the MLP (mlp plus normalization).
    """
    def __init__(
        self,
        head_count: int,
        embedding_len: int,
        mlp_hidden_size: int,
    ) -> None:
        super().__init__()
        self.embedding_len = embedding_len
        self.mlp_hidden_size = mlp_hidden_size
        self.head_count = head_count

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
        return self.mlp_residual(self.self_attention_residual(x))


class ViTEncoder(Module):
    """Encoder of the Vision Transformer ViT.

    Attributes:
        layer_count: The number of layers to apply.
        head_count: the number of head in multihead attention modules
        model: The stack of ViT layers.
    """

    def __init__(
        self,
        layer_count: int,
        head_count: int,
        embedding_len: int,
        mlp_hidden_size: int,
    ) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.head_count = head_count

        self.model = Sequential(
            {
                f"layer_{layer_idx}": ViTEncoderLayer(
                    head_count=head_count,
                    embedding_len=embedding_len,
                    mlp_hidden_size=mlp_hidden_size,
                )
                for layer_idx in range(layer_count)
            }
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ViTEmbedder(Module):
    """Embedder of the Vision Transformer ViT.

    Allow to pass from the image to the patch tokens (classification token included). Each patch
    token has a learnable positional encoding included.

    Attributes:
        patch_size: The size of the patch to use.
        image_width: The width of the image.
        image_height: The height of the image.
        embedding_len: The length of the embedding.
        linear: The linear layer to apply to the patches.
        positional_encoding: The positional encoding to apply to the patches.
        cls_token: The token to use for classification.
    """

    def __init__(
        self,
        patch_size: int,
        image_width: int,
        image_height: int,
        embedding_len: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embedding_len = embedding_len
        self.image_width = image_width
        self.image_height = image_height

        self.linear = Linear(
            in_features=patch_size * patch_size * 3,
            out_features=embedding_len,
        )

        self.positional_encoding = LearnablePositionalEncoding1D(
            sequence_len=(image_width // patch_size) * (image_height // patch_size) + 1,
            embedding_len=embedding_len,
        )
        self.cls_token = Parameter(randn(1, 1, embedding_len))

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = cat((cls_tokens, x), dim=1)
        return self.positional_encoding(x)


class ViT(Module):
    """Vintage implementation of the ViT model.

    See the paper_review.md file for more information.
    """

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
            image_width=self.patch_converter.final_width,
            image_height=self.patch_converter.final_height,
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
        classified = self.classification_mlp(encoded[:, 0, :])
        return softmax(classified, dim=-1)

    def __str__(self):
        return (
            f"vit{self.patch_converter.patch_size}-embed{self.embedder.embedding_len}"
            f"-mlp{self.classification_mlp.hidden_size}-"
            f"heads{self.encoder.head_count}-"
            f"layers{self.encoder.layer_count}"
        )
