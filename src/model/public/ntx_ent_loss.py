# Contains modifications from the lightly.ai source code found at:
# https://github.com/lightly-ai/lightly/blob/master/lightly/loss/ntx_ent_loss.py
# because theirs has a lot of bugs.
""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn.functional as F
from lightly.loss.memory_bank import MemoryBankModule
from lightly.utils import dist
from torch import nn


def contrastive_loss(x0, x1, tau, norm=True):
    # https://github.com/google-research/simclr/blob/master/objective.py
    bsize = x0.shape[0]
    target = torch.arange(bsize, device=x1.device)
    eye_mask = torch.eye(bsize, device=x1.device) * 1e9
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits11 = x1 @ x1.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    logits10 = x1 @ x0.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2


def lie_nt_xent_loss(out_1, out_2, out_3=None, temperature=0.07, mse=False, eps=1e-6):
    """
    DOES NOT assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    out_3: [batch_size, dim]
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    # out_3_dist: [batch_size * world_size, dim]
    # out: [2 * batch_size, dim]
    # out_dist: [3 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    if out_3 is not None:
        out_dist = torch.cat([out_1, out_2, out_3], dim=0)
    else:
        out_dist = torch.cat([out_1, out_2], dim=0)

    # cov and sim: [2 * batch_size, 3 * batch_size * world_size]
    # neg: [2 * batch_size]
    if mse:
        cov = -((out.unsqueeze(1) - out_dist.unsqueeze(0))**2).mean(dim=-1)
    else:
        cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    row_sub = torch.exp(torch.norm(out, dim=-1) / temperature)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    if mse:
        pos = -((out_1 - out_2)**2).mean(dim=-1)
        pos = torch.exp(pos / temperature)
    else:
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / (neg + eps)).mean()
    if loss < 0.0:
        print("Lie Contrastive loss can't be negative")
        raise ValueError("Lie Contrastive loss can't be negative")
    return loss


class NTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.
    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.
    - [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    - [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation. This flag has no effect if memory_bank_size > 0.
    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.
    Examples:
        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)
    """

    def __init__(
        self,
        temperature: float = 0.5,
        memory_bank_size: int = 0,
        gather_distributed: bool = False,
        reduction: str = "mean",
        normalize: bool = True,
        loss_type: str = "cos",
        detach_off_logit: bool = False,
    ):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.eps = 1e-8
        self.normalize = normalize
        self.loss_type = loss_type
        self.reduction = reduction
        self.detach_off_logit = detach_off_logit

        if abs(self.temperature) < self.eps:
            raise ValueError("Illegal temperature: abs({}) < 1e-8".format(self.temperature))

    def compute_loss(self, tensor0: torch.Tensor, tensor1: torch.Tensor):
        if self.loss_type == "cos":
            return torch.einsum("nc,mc->nm", tensor0, tensor1)
        elif self.loss_type == "mse":
            mse = ((tensor1.unsqueeze(1) - tensor0.unsqueeze(0)) ** 2).mean(dim=-1)
            return -mse
        else:
            raise NotImplementedError

    # Public implementation take from
    # https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/simclr.py
    def nt_xent(self, x0, x1):
        t = self.temperature
        x = torch.concat((x0, x1), dim=0)
        x = F.normalize(x, dim=1)
        x_scores = (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
        x_scale = x_scores / t  # scale with temperature

        # (2N-1)-way softmax without the score of i-th entry itself.
        # Set the diagonals to be large negative values, which become zeros after softmax.
        x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

        # targets 2N elements.
        targets = torch.arange(x.size()[0])
        targets[: len(targets) // 2] += len(x0)  # target of 2k element is 2k+1
        targets[len(targets) // 2 :] -= len(x0)  # target of 2k+1 element is 2k
        return F.cross_entropy(x_scale, targets.long().to(x_scale.device))

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        """Forward pass through Contrastive Cross-Entropy Loss.
        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.
        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
        Returns:
            Contrastive Cross Entropy Loss value.
        """

        device = out0.device
        batch_size, _ = out0.shape

        if self.normalize:
            # normalize the output to length 1
            out0 = nn.functional.normalize(out0, dim=1)
            out1 = nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = super(NTXentLoss, self).forward(out1, update=out0.requires_grad)

        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # use negatives from memory bank
            negatives = negatives.to(device)

            if self.loss_type == "cos":
                # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
                # of the i-th sample in the batch to its positive pair
                sim_pos = torch.einsum("nc,nc->n", out0, out1).unsqueeze(-1)
                # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
                # of the i-th sample to the j-th negative sample
                sim_neg = torch.einsum("nc,ck->nk", out0, negatives)
            elif self.loss_type == "mse":
                sim_pos = ((out0 - out1) ** 2).mean(dim=-1)
                sim_neg = self.compute_loss(out0, negatives.T)
            else:
                raise NotImplementedError

            # set the labels to the first "class", i.e. sim_pos,
            # so that it is maximized in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)

        else:
            # user other samples from batch as negatives
            # and create diagonal mask that only selects similarities between
            # views of the same image
            if self.gather_distributed and dist.world_size() > 1:
                # gather hidden representations from other processes
                out0_large = torch.cat(dist.gather(out0), 0)
                out1_large = torch.cat(dist.gather(out1), 0)
                diag_mask = dist.eye_rank(batch_size, device=out0.device)
            else:
                # single process
                out0_large = out0
                out1_large = out1
                diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

            # calculate similiarities
            # here n = batch_size and m = batch_size * world_size
            # the resulting vectors have shape (n, m)
            logits_00 = self.compute_loss(out0, out0_large) / self.temperature
            logits_01 = self.compute_loss(out0, out1_large) / self.temperature
            logits_10 = self.compute_loss(out1, out0_large) / self.temperature
            logits_11 = self.compute_loss(out1, out1_large) / self.temperature

            if self.detach_off_logit:
                logits_01 = logits_01.detach()
                logits_10 = logits_10.detach()

            # remove simliarities between same views of the same image
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)

            # concatenate logits
            # the logits tensor in the end has shape (2*n, 2*m-1)
            logits_0100 = torch.cat([logits_01, logits_00], dim=1)
            logits_1011 = torch.cat([logits_10, logits_11], dim=1)
            logits = torch.cat([logits_0100, logits_1011], dim=0)

            # create labels
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            labels = labels + dist.rank() * batch_size
            labels = labels.repeat(2)

        loss = self.cross_entropy(logits, labels)

        return loss
