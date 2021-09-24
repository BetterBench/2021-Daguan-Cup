import torch.nn as nn
def init_network(model, method='kaiming', exclude='embedding', seed=123):  # method='kaiming'
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name and w.ndim > 1:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                nn.init.constant_(w, 0)
