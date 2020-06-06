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

        self.alpha_w = nn.Parameter(torch.zeros(1))
        # self.gamma_w = nn.Parameter(torch.zeros(1))

        if bias:
            self.alpha_b = nn.Parameter(torch.zeros(1))
            # self.gamma_b = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        # Binarize weights
        binary_w = self.alpha_w * scaled_sign(self.weight)

        # Binarize bias if present
        binary_b = None
        if self.bias is not None:
            binary_b = self.alpha_b * scaled_sign(self.bias)

        return F.linear(inputs, binary_w, binary_b)

    def init_binary_weights(self):
        with torch.no_grad():
            # self.gamma_w.data = self.weight.mean().unsqueeze(0)
            self.alpha_w.data = torch.norm(self.weight, p=1).unsqueeze(0) / (self.in_features * self.out_features)

            if self.bias is not None:
                # self.gamma_b.data = self.bias.mean().unsqueeze(0)
                self.alpha_b.data = torch.norm(self.bias, p=1).unsqueeze(0) / self.out_features


class MultiScaleBinaryLinear(nn.Linear):
    def __init__(self, in_features, out_segments, output_features_per_segment, bias=True):
        super().__init__(in_features, out_segments * output_features_per_segment, bias)

        self.out_segments = out_segments
        self.output_features_per_segment = output_features_per_segment

        self.alpha_w = nn.Parameter(torch.zeros(out_segments))
        # self.gamma_w = nn.Parameter(torch.zeros(out_segments))

        if bias:
            self.alpha_b = nn.Parameter(torch.zeros(out_segments))
            # self.gamma_b = nn.Parameter(torch.zeros(out_segments))

    def forward(self, inputs):
        # Binarize weights with a different scaling factor per segment
        weight_segments = self.weight.view(self.out_segments, self.output_features_per_segment, self.in_features)
        alpha_w = self.alpha_w.view(-1, 1, 1)
        # gamma_w = self.gamma_w.view(-1, 1, 1)
        binary_w = (alpha_w * scaled_sign(weight_segments)).view_as(self.weight)

        # Binarize bias (if existent) with a different scaling factor per segment
        binary_b = None
        if self.bias is not None:
            bias_segments = self.bias.view(self.out_segments, self.output_features_per_segment)
            alpha_b = self.alpha_b.view(-1, 1)
            # gamma_b = self.gamma_b.view(-1, 1)
            binary_b = (alpha_b * scaled_sign(bias_segments)).view_as(self.bias)

        return F.linear(inputs, binary_w, binary_b)

    def init_binary_weights(self):
        with torch.no_grad():
            weight_segments = self.weight.view(self.out_segments, self.output_features_per_segment, self.in_features)
            # self.gamma_w.data = weight_segments.mean(dim=(1, 2))
            # gamma_w = self.gamma_w.view(-1, 1, 1)
            self.alpha_w.data = torch.norm(weight_segments, p=1, dim=(1, 2)) / (self.output_features_per_segment * self.in_features)
            if self.bias is not None:
                bias_segments = self.bias.view(self.out_segments, self.output_features_per_segment)
                # self.gamma_b.data = bias_segments.mean(dim=-1)
                # gamma_b = self.gamma_b.view(-1, 1)
                self.alpha_b.data = torch.norm(bias_segments, dim=-1, p=1) / self.output_features_per_segment


class BinaryEmbeddingSingle(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

        self.alpha_w = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        binary_w = self.alpha_w * scaled_sign(self.weight)

        return F.embedding(inputs, binary_w, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

    def init_binary_weights(self):
        with torch.no_grad():
            self.alpha_w.data = torch.norm(self.weight, p=1).unsqueeze(0) / (self.num_embeddings * self.embedding_dim)


class BinaryEmbeddingH(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

        self.alpha_w = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, input):
        alpha_w = self.alpha_w.view(1, -1)
        binary_w = alpha_w * scaled_sign(self.weight)

        return F.embedding(input, binary_w, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

    def init_binary_weights(self):
        with torch.no_grad():
            self.alpha_w.data = torch.norm(self.weight, p=1, dim=0) / self.num_embeddings


class BinaryEmbeddingW(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

        self.alpha_w = nn.Parameter(torch.zeros(num_embeddings))

    def forward(self, input):
        alpha_w = self.alpha_w.view(-1, 1)
        binary_w = alpha_w * scaled_sign(self.weight)

        return F.embedding(input, binary_w, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

    def init_binary_weights(self):
        with torch.no_grad():
            self.alpha_w.data = torch.norm(self.weight, p=0, dim=1) / self.embedding_dim


# class BinaryEmbeddingSVD(nn.Embedding):
#     def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
#         super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
#
#         self.alpha_w = nn.Parameter(torch.zeros(num_embeddings))
#         self.beta_w = nn.Parameter(torch.zeros(embedding_dim))
#
#     def forward(self, input):
#         binary_w_no_sign = torch.ger(self.alpha_w, self.beta_w)
#         binary_w = binary_w_no_sign * scaled_sign(self.weight)
#
#         return F.embedding(input, binary_w, self.padding_idx, self.max_norm,
#                            self.norm_type, self.scale_grad_by_freq, self.sparse)
#
#     def init_binary_weights(self):
#         with torch.no_grad():
#             U, S, V = torch.svd(torch.abs(self.weight))
#
#             self.alpha_w.data = U[:, 0]
#             self.beta_w.data = S[0] * V[:, 0]
