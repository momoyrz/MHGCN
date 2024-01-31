import torch
from timm.models.registry import register_model
from torch import nn

import torch.nn.functional as F

from models.layers import WHGCN_Conv

class MHGCN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, num_layers, cosine_threshold, Euclidean_threshold, cosine_k, gaussian_gamma,
                 gaussian_threshold, gaussian_k, num_node, num_hyperedge, scaler, kmeans_k, dropout=0.5, residual=True):
        super(MHGCN, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.cosine_threshold = cosine_threshold
        self.gaussian_gamma = gaussian_gamma
        self.cosine_k = cosine_k
        self.Euclidean_threshold = Euclidean_threshold
        self.gaussian_threshold = gaussian_threshold
        self.gaussian_k = gaussian_k
        self.num_node = num_node
        self.num_hyperedge = num_hyperedge
        self.scaler = scaler
        self.kmeans_k = kmeans_k

        self.layers = nn.ModuleList([WHGCN_Conv(cosine_threshold, cosine_k, Euclidean_threshold, gaussian_gamma, gaussian_threshold,
                                                gaussian_k, kmeans_k, num_node, num_hyperedge, in_ch, n_hid, scaler)])

        # Initialize the middle layers
        for _ in range(num_layers - 2):
            self.layers.append(WHGCN_Conv(cosine_threshold, cosine_k, Euclidean_threshold, gaussian_gamma, gaussian_threshold,
                                          gaussian_k, kmeans_k, num_node, num_hyperedge, n_hid, n_hid, scaler))

        # Initialize the last layer
        self.layers.append(WHGCN_Conv(cosine_threshold, cosine_k, Euclidean_threshold, gaussian_gamma, gaussian_threshold,
                                      gaussian_k, kmeans_k, num_node, num_hyperedge, n_hid, n_class, scaler))

    def forward(self, x, H):
        for i, layer in enumerate(self.layers):
            if self.residual and i != 0 and i != len(self.layers) - 1:  # Apply residual connections to middle layers
                x_res = x
                x = F.relu(layer(x, H))
                x += x_res
            else:
                x = F.relu(layer(x, H))

            if i != len(self.layers) - 1:  # Apply dropout to all layers except the last one
                x = F.dropout(x, self.dropout)

        return x

    @torch.no_grad()
    def feature_extract(self, x, H):
        for i, layer in enumerate(self.layers):
            if self.residual and i != 0:
                x_res = x
                x = F.relu(layer(x, H))
                x += x_res
            else:
                x = F.relu(layer(x, H))

            if i != len(self.layers) - 1:  # Apply dropout to all layers except the last one
                x = F.dropout(x, self.dropout)
            if i == len(self.layers) - 2:
                return x
        return x


@register_model
def dhgnn(**kwargs):
    return MHGCN(**kwargs)
