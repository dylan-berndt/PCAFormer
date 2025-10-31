from .config import *
import torch
import torch.nn as nn
import os
import copy
import time
from torchvision.models import swin_v2_t, resnet50


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
        self.transformer = nn.TransformerEncoderLayer(**config.transformer)

    def forward(self, x):
        if x.shape[1] != self.config.k:
            x = torch.transpose(x, 1, 2)
            u, s, v = torch.pca_lowrank(x, center=True, q=self.config.k)
            x = torch.matmul(x, v[:, :, :self.config.k])
            x = torch.transpose(x, 1, 2)

        x = self.transformer(x)

        return x
    

class PCAFormer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.patching = PatchEmbedding(config)

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
        
        self.classification = nn.Sequential(
            nn.Linear(config.embedDim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.patching(x)

        x = self.transformer(x)

        x = torch.mean(x, dim=1)

        x = self.classification(x)

        return x
    

class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = resnet50(weights=None)

        features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

class SwinTransformerV2Tiny(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = swin_v2_t(weights=None)

        features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

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

    
