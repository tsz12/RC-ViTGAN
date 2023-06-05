from importlib import import_module

from training.gan.aug_both import loss_G_pre_fn


def setup(P):
    mod = import_module(f'.{P.mode}', 'training.gan')
    #setup需要根据不同项目的配置，导入对应的配置文件进行运行（导入配置文件的方法或者类）
    #这里P.mode如果是aug_both的话，这里相当于导入了training.gan.aug_both.py
    loss_G_fn = mod.loss_G_fn
    loss_D_fn = mod.loss_D_fn
    loss_G_pre_fn=mod.loss_G_pre_fn#我加的！！
    loss_train1_fn=mod.loss_train1_fn
    loss_train1_AB_fn=mod.loss_train1_AB_fn
    loss_train2_fn=mod.loss_train2_fn
    loss_D_my_fn=mod.loss_D_my_fn
    loss_G_my_fn=mod.loss_G_my_fn
    loss_D_my_match_fn=mod.loss_D_my_match_fn
    loss_G_my_match_fn=mod.loss_G_my_match_fn
    loss_G_my_match_wochloss_fn=mod.loss_G_my_match_wochloss_fn
    loss_G_my_match_wotargetloss_fn=mod.loss_G_my_match_wotargetloss_fn
    if P.mode == 'std':
        filename = f"{P.mode}_{P.penalty}"
        if 'cr' in P.penalty:
            filename += f'_{P.aug}'
    elif P.mode == 'aug':
        filename = f"{P.mode}_{P.aug}_{P.penalty}"
    elif P.mode == 'aug_both':
        filename = f"{P.mode}_{P.aug}_{P.penalty}"
    elif P.mode == 'simclr_only':
        filename = f"{P.mode}_{P.aug}_T{P.temp}"
    elif P.mode == 'contrad':
        filename = f"{P.mode}_{P.aug}_L{P.lbd_a}_T{P.temp}"
    else:
        raise NotImplementedError()

    P.filename = filename
    P.train_fn = {
        "G": loss_G_fn,
        "D": loss_D_fn,
        "G_pre":loss_G_pre_fn,#!!!!
        "train1":loss_train1_fn,
        "train1_AB":loss_train1_AB_fn,
        "train2":loss_train2_fn,
        "train3_D":loss_D_my_fn,
        "train3_G":loss_G_my_fn,
        "train3_D_match":loss_D_my_match_fn,
        "train3_G_match":loss_G_my_match_fn,
        "train3_G_wochloss_match":loss_G_my_match_wochloss_fn,
        "train3_G_wotargetloss_match":loss_G_my_match_wotargetloss_fn,
    }
    return P