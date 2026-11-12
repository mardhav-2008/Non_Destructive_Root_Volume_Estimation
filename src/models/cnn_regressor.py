import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(2, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.25),
        )

    def forward(self, image):
        x = self.features(image)
        x = self.pool(x)
        return self.output(x)


class MetadataEncoder(nn.Module):
    def __init__(self, category_sizes, numeric_features=3):
        super().__init__()

        # Embedding dimensions for:
        # PlantNumber, Genotype, Stage, Side
        embedding_dims = [4, 4, 2, 2]

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(size + 1, dim)
                for size, dim in zip(category_sizes, embedding_dims)
            ]
        )

        total_embedding_dim = sum(embedding_dims)

        self.network = nn.Sequential(
            nn.Linear(
                total_embedding_dim + numeric_features,
                32,
            ),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 32),
            nn.GELU(),
        )

    def forward(self, categorical, numeric):
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]

        x = torch.cat(embedded + [numeric], dim=1)
        return self.network(x)


class RootVolumeRegressor(nn.Module):
    def __init__(self, category_sizes, mode="multimodal"):
        super().__init__()

        if mode not in {"image", "metadata", "multimodal"}:
            raise ValueError(f"Unknown model mode: {mode}")

        self.mode = mode

        if mode in {"image", "multimodal"}:
            self.image_encoder = ImageEncoder()

        if mode in {"metadata", "multimodal"}:
            self.metadata_encoder = MetadataEncoder(category_sizes)

        if mode == "image":
            head_input = 128

        elif mode == "metadata":
            head_input = 32

        else:
            head_input = 128 + 32

        self.regression_head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, image=None, categorical=None, numeric=None):
        features = []

        if self.mode in {"image", "multimodal"}:
            features.append(self.image_encoder(image))

        if self.mode in {"metadata", "multimodal"}:
            features.append(self.metadata_encoder(categorical, numeric))

        combined = torch.cat(features, dim=1)

        return self.regression_head(combined).squeeze(1)
