import torch
import torch.nn.functional as F

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    z = torch.cat([z1, z2], dim=0)  # 2B
    sim = torch.matmul(z, z.transpose(0, 1))  # 2B x 2B

    sim = sim / 0.5

    logits = torch.tril(sim, diagonal=-1)[:, :-1]
    logits += torch.triu(sim, diagonal=1)[:, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[i, B + i - 1].mean() + logits[B + i, i].mean()) / 2
    return loss
