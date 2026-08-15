import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=None, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.num_classes = num_classes

        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        
        if not pretrained:
            import torchvision.models as models
            weights_mapping = {
                18: models.ResNet18_Weights.DEFAULT,
                34: models.ResNet34_Weights.DEFAULT,
                50: models.ResNet50_Weights.DEFAULT,
                101: models.ResNet101_Weights.DEFAULT,
                152: models.ResNet152_Weights.DEFAULT,
            }
            self.base = ResNet.__factory[depth](weights=weights_mapping[depth])
        else:
            self.base = ResNet.__factory[depth](weights=None)
            self.base.load_state_dict(torch.load(pretrained))

        for mo in self.base.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        
        out_planes = self.base.fc.in_features
        self.bn = nn.BatchNorm1d(out_planes)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)

        if self.num_classes > 0:
            self.classifier = nn.Linear(out_planes, self.num_classes)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if output_feature == 'pool5':
            x = F.normalize(x)
            return x

        xx = self.bn(x)
        l2feat = F.normalize(xx)
        if output_feature == 'l2feat':
            return l2feat

        if self.num_classes > 0:
            xx = self.classifier(xx)
        return x, xx

def resnet18(**kwargs):
    return ResNet(18, **kwargs)

def resnet34(**kwargs):
    return ResNet(34, **kwargs)

def resnet50(**kwargs):
    return ResNet(50, **kwargs)

def resnet101(**kwargs):
    return ResNet(101, **kwargs)

def resnet152(**kwargs):
    return ResNet(152, **kwargs)