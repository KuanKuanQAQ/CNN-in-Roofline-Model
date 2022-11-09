import csv
import torch
import torch.nn as nn
from torchvision.models import *

from cout_flops import profile, clever_format

model_dict = {
    'alexnet': alexnet,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'inception_v3': inception_v3,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
    'mobilenet_v2': mobilenet_v2,
    'mobilenet_v3_large': mobilenet_v3_large,
    'mobilenet_v3_small': mobilenet_v3_small,
}
header = ['model_name', 'FLOPs', 'params','mem','intensity', 'forward_time']

f = open('results/csv_file.csv', 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
# write the header
writer.writerow(header)

for name, model in model_dict.items():
    if name == 'inception_v3':
        net = model(init_weights=True).cuda()
    else:
        net = model().cuda()
    avg_forward_time = 0
    for i in range(10):
        input = torch.randn(1, 3, 224, 224).cuda()

        flops, params, mem, forward_time = profile(net, inputs=(input, ))
        intensity = flops/(mem*4)
        flops, params, mem = clever_format([flops, params, mem*4], "%.3f")
        avg_forward_time += forward_time
        if i == 0:
            continue
        if i == 9:
            print(name, flops, params, mem, round(intensity,2), str(round(avg_forward_time/9,3))+'s')
            writer.writerow([name, flops, params, mem, intensity, round(avg_forward_time/9,3)])
            break
'''
@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)
    if type(m) == nn.Conv2d:
        print(m.weight.shape)
    if type(m) == nn.BatchNorm2d:
        print(getattr(m))
net = nn.Sequential(nn.Conv2d(4, 2, 3), nn.BatchNorm2d(2), nn.Linear(2, 2))
net.apply(init_weights)
'''