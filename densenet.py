import math
import torch
import torch.nn as nn
import torch.nn.functional as f


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, droprate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = droprate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = f.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, droprate: float = 0.0):
        super().__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = droprate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = f.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = f.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, droprate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = droprate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = f.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return f.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers: int, in_planes: int, growth_rate: int,
                 block, droprate: float = 0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, droprate)

    @staticmethod
    def _make_layer(block, in_planes: int,
                    growth_rate: int, nb_layers: int, droprate: float) -> nn.Sequential:
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, droprate))
        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(self, depth: int, num_classes: int, growth_rate: int = 12,
                 reduction: float = 0.5, bottleneck: bool = True, droprate: float = 0.0):
        super().__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = in_planes + n * growth_rate
        self.trans1 = TransitionBlock(in_planes, math.floor(in_planes * reduction), droprate=droprate)
        in_planes = math.floor(in_planes * reduction)
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = in_planes + n * growth_rate
        self.trans2 = TransitionBlock(in_planes, math.floor(in_planes * reduction), droprate=droprate)
        in_planes = math.floor(in_planes * reduction)
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = in_planes + n * growth_rate
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = f.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


class DenseNet4(nn.Module):
    def __init__(self, depth: int, num_classes: int, growth_rate: int = 12,
                 reduction: float = 0.5, bottleneck: bool = True, droprate: float = 0.0):
        super().__init__()
        in_planes = 2 * growth_rate

        if depth == 121:
            stages = [6, 12, 24, 16]
        elif depth == 161:
            stages = [6, 12, 36, 24]
        elif depth == 169:
            stages = [6, 12, 32, 32]
        elif depth == 201:
            stages = [6, 12, 48, 32]
        else:
            n = (depth - 4) / 3
            stages = [int(n / 2) if bottleneck else int(n)
                      for _ in range(4)]

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        # 1st trans before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 1st block
        self.block1 = DenseBlock(stages[0], in_planes, growth_rate, block, droprate)
        in_planes = in_planes + growth_rate
        self.trans1 = TransitionBlock(in_planes, math.floor(in_planes * reduction), droprate=droprate)
        in_planes = math.floor(in_planes * reduction)
        # 2nd block
        self.block2 = DenseBlock(stages[1], in_planes, growth_rate, block, droprate)
        in_planes = in_planes + growth_rate
        self.trans2 = TransitionBlock(in_planes, math.floor(in_planes * reduction), droprate=droprate)
        in_planes = math.floor(in_planes * reduction)
        # 3rd block
        self.block3 = DenseBlock(stages[2], in_planes, growth_rate, block, droprate)
        in_planes = in_planes + growth_rate
        self.trans3 = TransitionBlock(in_planes, math.floor(in_planes * reduction), droprate=droprate)
        in_planes = math.floor(in_planes * reduction)
        # 4th block
        self.block4 = DenseBlock(stages[3], in_planes, growth_rate, block, droprate)
        in_planes = in_planes + growth_rate
        # global average pooling and classifier
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x) -> torch.Tensor:
        out = self.bn1(self.conv1(x))
        out = self.maxp(self.relu1(out))
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.trans3(self.block3(out))
        out = self.block4(out)
        out = self.relu2(self.bn2(out))
        out = f.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
