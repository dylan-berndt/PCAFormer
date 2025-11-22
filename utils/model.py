from .config import *
import torch
import torch.nn as nn
import os
import copy
import time
from torchvision.models import swin_v2_t, resnet50, ResNet50_Weights, Swin_V2_T_Weights, VisionTransformer


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=config.embedDim,
            kernel_size=config.patchSize,
            stride=config.patchSize
        )

        numPatches = (config.imageSize // config.patchSize) ** 2
        self.positional = nn.Parameter(torch.zeros(1, numPatches, config.embedDim))

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional[:, :x.size(1), :]
        return x


class PCAFormerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.ln1 = nn.LayerNorm(config.transformer.d_model)
        self.attn = nn.MultiheadAttention(config.transformer.d_model, config.transformer.nhead,
                                          batch_first=config.transformer.batch_first)
        self.dropout = nn.Dropout(config.transformer.dropout)

        self.ln2 = nn.LayerNorm(config.transformer.d_model)

        self.mlp = nn.Sequential(
            nn.Linear(config.transformer.d_model, config.transformer.dim_feedforward),
            nn.LayerNorm(config.transformer.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.transformer.dropout),
            nn.Linear(config.transformer.dim_feedforward, config.transformer.d_model),
            nn.Dropout(config.transformer.dropout)
        )

    def forward(self, x):
        if x.shape[1] != self.config.k + 1:
            with torch.no_grad():
                x2 = x[:, 1:]
                x2 = torch.transpose(x2, 1, 2)
                # std = x.std(dim=-1, keepdim=True)
                # x = (x - x.mean(dim=-1, keepdim=True)) / std
                x1 = torch.zeros([x2.shape[0], x2.shape[1], self.config.k])
                for b in range(x2.shape[0]):
                    u, s, v = torch.pca_lowrank(x2[b], center=True, q=self.config.k)
                    x1[b] = torch.matmul(x2[b], v[:, :self.config.k])
                # u, s, v = torch.pca_lowrank(x2, center=True, q=self.config.k)
                # x1 = torch.matmul(x2, v[:, :, :self.config.k])
                # x = x * std
                x2 = torch.transpose(x1, 1, 2)
                x = torch.cat([x[:, 0].unsqueeze(1), x2], dim=1)

        i = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + i

        y = self.ln2(x)
        y = self.mlp(y)

        return x + y
    

class PCAFormer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.patching = PatchEmbedding(config)

        self.classToken = nn.Parameter(torch.zeros(1, 1, config.embedDim))

        self.layers = nn.ModuleList([])

        k = (config.imageSize // config.patchSize) ** 2
        for l in range(config.layers):
            layerConfig = copy.deepcopy(config.layer)
            layerConfig["k"] = k

            layer = PCAFormerLayer(layerConfig)
            self.layers.append(layer)

            if l % config.patchLayerStride == 0:
                k = k // config.patchLayerCompression

        self.transformer = nn.Sequential(*self.layers)
        
        self.classification = nn.Linear(config.embedDim, config.classes)
        
    def forward(self, x):
        x = self.patching(x)

        classToken = self.classToken.expand(x.shape[0], -1, -1)
        x = torch.cat([classToken, x], dim=1)

        x = self.transformer(x)

        x = x[:, 0]

        x = self.classification(x)

        return x
    

class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = resnet50(weights=None)

        features = self.model.fc.in_features
        self.model.fc = nn.Linear(features, config.classes)

    def forward(self, x):
        return self.model(x)
    

class SwinTransformerV2Tiny(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = swin_v2_t(weights=None)

        features = self.model.head.in_features
        self.model.head = nn.Linear(features, config.classes)

    def forward(self, x):
        return self.model(x)


class VisionTransformerWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.m = VisionTransformer(256, 16, 12, 8, 128, 512)

    def forward(self, x):
        return self.m(x)
    

def testModel(model, samples, batchSize):
    start = time.time()
    for i in range(samples):
        inputs = torch.rand(batchSize, 3, 256, 256)
        outputs = model(inputs)

        print(f"\r{i + 1}/{256}", end="")

    print()

    end = time.time()
    seconds = end - start
    print(f"{seconds:.2f} Total seconds | {(samples * batchSize) / seconds:.2f} Samples per second")

        
if __name__ == "__main__":
    config = Config().load(os.path.join("configs", "config.json")).model
    standard = Config().load(os.path.join("configs", "standard.json")).model

    m1 = ResNet50(config)
    m2 = PCAFormer(standard)
    m3 = SwinTransformerV2Tiny(config)
    m4 = PCAFormer(config)

    samples = 256
    batchSize = 32

    print(f"\nResNet50 | {sum([p.numel() for p in m1.parameters()])} parameters")
    testModel(m1, samples, batchSize)
    print(f"\nStandard ViT | {sum([p.numel() for p in m2.parameters()])} parameters")
    testModel(m2, samples, batchSize)
    print(f"\nSwin Tiny | {sum([p.numel() for p in m3.parameters()])} parameters")
    testModel(m3, samples, batchSize)
    print(f"\nPCAFormer | {sum([p.numel() for p in m4.parameters()])} parameters")
    testModel(m4, samples, batchSize)

    
