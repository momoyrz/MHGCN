import math

import numpy as np

def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    """Cosine scheduler.
    :param base_value: base learning rate.
    :param final_value: final learning rate.
    :param epochs: total epochs.
    :param warmup_epochs: warmup epochs.
    :param start_warmup_value: start warmup learning rate.
    """
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs)   # 预热步骤的学习率

    iters = np.arange(epochs - warmup_epochs)  # 除去预热步骤的迭代次数
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * t / len(iters))) for t in iters]
    )  # 余弦退火学习率

    schedule = np.concatenate((warmup_schedule, schedule))  # 将预热步骤和余弦退火步骤拼接起来
    return schedule
