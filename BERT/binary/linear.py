import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        sgn = inputs.clone()
        sgn[inputs < 0.] = -1.
        sgn[inputs >= 0.] = 1.
        return sgn

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, = ctx.saved_tensors
        grad = inputs.clone()
        grad[torch.abs(inputs) > 1.] = 0
        return torch.mul(grad, grad_outputs)


scaled_sign = ScaledSign.apply


class BinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, inputs):
        # return F.linear(inputs, self.weight, self.bias)  # uncomment for original linear layer

        # Binarize weights
        alpha_w = torch.norm(self.weight, p=1) / (self.in_features * self.out_features)
        binary_w = alpha_w * scaled_sign(self.weight)

        # Binarize bias if present
        binary_b = None
        if self.bias is not None:
            alpha_b = torch.norm(self.bias, p=1) / self.out_features
            binary_b = alpha_b * scaled_sign(self.bias)

        return F.linear(inputs, binary_w, binary_b)

