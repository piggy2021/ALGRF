import torch
from torch import nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        net = models.resnet50(pretrained=True)
        # net.load_state_dict(torch.load(resnext101_32_path))

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:4])
        self.layer1 = net[4]
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4

if __name__ == '__main__':
    model = ResNet50()