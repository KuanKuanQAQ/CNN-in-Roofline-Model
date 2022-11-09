import torch
import numpy as np


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

def l_sum(in_list):
    res = 0
    for _ in in_list:
        res += _
    return res


def calculate_parameters(param_list):
    total_params = 0
    for p in param_list:
        total_params += torch.DoubleTensor([p.nelement()])
    return total_params


def calculate_zero_ops():
    return torch.DoubleTensor([int(0)])

def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    # n, out_c, oh, ow = output_size
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * ((in_c // g) * l_prod(kernel_size[2:]) + in_c // g)

def calculate_norm(input_size):
    """input is a number not a array or tensor"""
    return torch.DoubleTensor([2 * input_size])

def calculate_relu_flops(input_size):
    # x[x < 0] = 0
    return 0
    

def calculate_relu(input: torch.Tensor):
    return torch.DoubleTensor([int((input<0).sum())])


def calculate_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])


def calculate_avgpool(input_size):
    return torch.DoubleTensor([int(input_size)])


def calculate_adaptive_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])

def calculate_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(2 * in_feature * num_elements)])


def counter_matmul(input_size, output_size):
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(input_size) * output_size[-1]


def counter_mul(input_size):
    return input_size


def counter_pow(input_size):
    return input_size


def counter_sqrt(input_size):
    return input_size


def counter_div(input_size):
    return input_size
