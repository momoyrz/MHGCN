import torch.nn

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="none"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = torch.nn.CrossEntropyLoss()


    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        p = torch.exp(-ce_loss)

        if self.alpha >= 0:
            loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss