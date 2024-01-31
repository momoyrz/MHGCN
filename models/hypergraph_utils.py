import torch


def compute_cosine_similarity(x):
    x_normalized = torch.nn.functional.normalize(x, p=2, dim=1)
    cosine_matrix = x_normalized @ x_normalized.T
    return cosine_matrix

def compute_Euclidean_similarity(x):
    norm = torch.sum(x ** 2, dim=1)
    sq_dists = -2 * x @ x.T + norm.unsqueeze(1) + norm.unsqueeze(0)
    Euclidean_dist = torch.sqrt(sq_dists)
    return Euclidean_dist

def compute_gaussian_similarity(x, gaussian_gamma):
    norm = torch.sum(x ** 2, dim=1)
    sq_dists = -2 * x @ x.T + norm.unsqueeze(1) + norm.unsqueeze(0)
    gaussian_dist = torch.exp(gaussian_gamma * sq_dists)
    return gaussian_dist


def sparse_matrix(H, matrix, threshold, k):
    if threshold > 0:
        matrix_threshold = (matrix > threshold).float()
        # 如果H中所有元素都为0
        if torch.sum(H) == 0:
            H = matrix_threshold
        else:
            H = torch.cat((H, matrix_threshold), dim=1)
    if k > 0:
        topk_values, topk_indices = torch.topk(matrix, k=k, dim=0)
        sparse_matrix = torch.zeros_like(matrix).scatter_(0, topk_indices, torch.ones_like(topk_values))
        if torch.sum(H) == 0:
            H = sparse_matrix
        else:
            H = torch.cat((H, sparse_matrix), dim=1)
    return H


def kmeans(x, k, max_iters=100, tol=1e-4):
    # 1. 随机选择 k 个质心
    centroids = x[torch.randperm(x.size(0))[:k]]

    for _ in range(max_iters):
        # 2. 计算每个点到质心的距离
        dist = torch.cdist(x, centroids)

        # 3. 将每个点分配给最近的质心
        labels = dist.argmin(dim=1)

        # 4. 计算新的质心
        new_centroids = torch.stack([x[labels == i].mean(0) for i in range(k)])

        # 5. 检查质心是否收敛
        if torch.all(torch.abs(centroids - new_centroids) < tol):
            break

        centroids = new_centroids

    return labels

def compute_kmeans_similarity(x, k):
    """
    :param x: 特征矩阵
    :param k: 聚类个数
    """
    labels = kmeans(x, k)
    H = torch.zeros((x.shape[0], k)).to(x.device)
    for i in range(x.shape[0]):
        H[i][labels[i]] = 1
    return H