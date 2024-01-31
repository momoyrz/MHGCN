import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch

from models.hypergraph_utils import compute_cosine_similarity, compute_gaussian_similarity, sparse_matrix, kmeans, \
    compute_kmeans_similarity, compute_Euclidean_similarity

class WHGCN_Conv(nn.Module):
    def __init__(self, cosine_threshold, cosine_k, Euclidean_threshold, gaussian_gamma, gaussian_threshold, gaussian_k, kmeans_k,
                 num_node, num_hyperedge, in_ft, out_ft, scaler):
        super(WHGCN_Conv, self).__init__()
        self.cosine_threshold = cosine_threshold
        self.gaussian_gamma = gaussian_gamma
        self.Euclidean_threshold = Euclidean_threshold
        self.cosine_k = cosine_k
        self.gaussian_threshold = gaussian_threshold
        self.gaussian_k = gaussian_k
        self.num_node = num_node
        self.num_hyperedge = num_hyperedge
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.scaler = scaler
        self.kmeans_k = kmeans_k
        self.proj = nn.Linear(self.in_ft, self.out_ft, bias=True)
        self.weight1 = Parameter(torch.Tensor(num_node, num_hyperedge))
        self.weight2 = Parameter(torch.Tensor(num_node, num_hyperedge))

        self.init_parameters()
    def forward(self, x, H):
        if self.cosine_threshold > 0 or self.cosine_k > 0:
            cosine_matrix = compute_cosine_similarity(x)
            H = sparse_matrix(H, cosine_matrix, self.cosine_threshold, self.cosine_k)
        if self.gaussian_threshold > 0 or self.gaussian_gamma > 0:
            gaussian_dist = compute_gaussian_similarity(x, self.gaussian_gamma)
            H = sparse_matrix(H, gaussian_dist, self.gaussian_threshold, self.gaussian_k)
        if self.Euclidean_threshold > 0:
            Euclidean_dist = compute_Euclidean_similarity(x)
            H = sparse_matrix(H, Euclidean_dist, self.Euclidean_threshold, 0)
        if self.kmeans_k > 0:
            kmeans_H = compute_kmeans_similarity(x, self.kmeans_k)
            H = torch.cat((H, kmeans_H), dim=1)
        x = self.proj(x)
        # 将结点特征聚合到超边上
        node_weights = (self.weight1 * H).T
        # 对node_weights的列进行归一化。每一列除以该列的和
        node_weights = torch.nn.functional.normalize(node_weights, p=1, dim=1) * self.scaler
        x = node_weights @ x
        # 将超边特征聚合到结点上
        hyperedge_weights = (self.weight2 * H)
        # 对hyperedge_weights的行进行归一化。每一行除以该行的和
        hyperedge_weights = torch.nn.functional.normalize(hyperedge_weights, p=1, dim=1) * self.scaler
        x = hyperedge_weights @ x
        return x

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(4))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(4))
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(4))
        # 将self.weight1和self.weight2中的负数变成相反数
        self.weight1.data = torch.abs(self.weight1.data)
        self.weight2.data = torch.abs(self.weight2.data)
        if self.proj.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.proj.bias, -bound, bound)
