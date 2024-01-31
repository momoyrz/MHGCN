import torch


def create_optimizer(args, model, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip)

    opt_args = dict(lr=args.base_lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    if opt_split[0] == 'sgd':
        optimizer = torch.optim.SGD(parameters, momentum=0.9, nesterov=(opt_lower == 'nesterov'), **opt_args)
    elif opt_split[0] == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_split[0] == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError('Optimizer {} not supported'.format(args.opt))

    return optimizer


def get_parameter_groups(model, weight_decay, skip_list=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():  # 遍历模型的所有参数
        if not param.requires_grad:  # 如果参数不需要梯度，则跳过
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            # 如果参数是一维的，或者是bias，或者在skip_list中，则不需要weight_decay.
            group_name = "no_decay"  # 不需要weight_decay
            this_weight_decay = 0.  # weight_decay为0
        else:
            group_name = "decay"  # 需要weight_decay
            this_weight_decay = weight_decay  # weight_decay为设置的值

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }

        parameter_group_vars[group_name]["params"].append(param)  # 将参数添加到对应的group中
        parameter_group_names[group_name]["params"].append(name)  # 将参数名添加到对应的group中

    return list(parameter_group_vars.values())