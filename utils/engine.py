import torch.nn

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, features: torch.Tensor, G: torch.Tensor,
                    labels: torch.Tensor, idx_train: torch.Tensor, idx_val: torch.Tensor,
                    idx_test: torch.Tensor, device: torch.device,
                    epoch: int, lr_schedule_values=None, logger=None,):
    model.train()
    model = model.to(device)
    optimizer.zero_grad()

    if lr_schedule_values is not None:
        lr = lr_schedule_values[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    features, G, labels = features.to(device), G.to(device), labels.to(device)
    idx_train, idx_val, idx_test = idx_train.to(device), idx_val.to(device), idx_test.to(device)

    output = model(features, G)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = output[idx_train].max(1)[1].eq(labels[idx_train]).sum().item() / len(idx_train)
    loss_train.backward()
    optimizer.step()

    output_formate = 'Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.4f}, LR: {:.4f}'.format(
        epoch, loss_train.item(), acc_train, lr)
    if logger is not None:
        logger.info(output_formate)

    return output

