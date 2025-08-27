import torch
import torch.nn as nn
from .config import NUM_CLASSES

# ---------- SimpleCNN (kept for reference/toggling) ----------
class SimpleCNN(nn.Module):
    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self.block(64, 64)     # 32→16
        self.stage2 = self.block(64, 128)    # 16→8
        self.stage3 = nn.Sequential(         # 8→4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return self.fc(x)

# ---------- ResNet for CIFAR (ResNet-18) ----------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNetCIFAR(nn.Module):
    """
    ResNet-18 variant for CIFAR-10:
    - First conv 3x3, stride 1 (no 7x7, no maxpool)
    - Stages: [2,2,2,2] blocks with downsampling at stage starts
    """
    def __init__(self, block=BasicBlock, layers=(2,2,2,2), num_classes=NUM_CLASSES, dropout=0.0):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)  # 32x32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 16x16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 8x8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # He init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# ---------- WideResNet (WRN-28-10), optional ----------
# Lightweight implementation tailored for CIFAR-10
class BasicBlockWRN(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.equalInOut = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.droprate = drop_rate
        self.shortcut = (None if self.equalInOut else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False))

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = nn.functional.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return out + (x if self.equalInOut else self.shortcut(x))

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    """WRN-28-10 for CIFAR (depth=28, widen_factor=10)"""
    def __init__(self, depth=28, widen_factor=10, num_classes=NUM_CLASSES, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlockWRN, 1, drop_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlockWRN, 2, drop_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlockWRN, 2, drop_rate)
        self.bn = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(nStages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.pool(x); x = torch.flatten(x, 1)
        return self.fc(x)

# ---------- Factory ----------
def build_model(name: str, num_classes=NUM_CLASSES, dropout=0.0):
    name = (name or "").lower()
    if name in ["simplecnn", "simple_cnn", "cnn"]:
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    if name in ["resnet18", "resnet18_cifar", "resnet_cifar"]:
        return ResNetCIFAR(num_classes=num_classes, dropout=dropout)
    if name in ["wrn28_10", "wideresnet", "wide_resnet"]:
        return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, drop_rate=dropout)
    raise ValueError(f"Unknown model_name: {name}")
