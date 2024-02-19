import os
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import faiss
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, args):
        super().__init__()
        self.temperature = args.moco_t
        self.base_temperature = args.moco_t

    def forward(self, features, mask=None, weight=None, batch_size=-1):
        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().cuda()
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).cuda(),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * weight *log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss


class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 计算未经加权的交叉熵损失
        if self.weight is not None:
            ce_loss = ce_loss * self.weight  # 对损失进行加权处理
        return torch.mean(ce_loss)


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger

def load_moco_model(moco_path, gpu_num, model, ema_model, requires_grad=False):
    state = torch.load(moco_path, map_location='cuda:' + str(gpu_num))['state_dict']
    model_paras = {k.replace('backbone', 'encoder'): v for k, v in state.items()}
    model_paras = {k.replace('projector', 'head'): v for k, v in model_paras.items()}
    model_paras = {k.replace('downsample', 'shortcut'): v for k, v in model_paras.items()}
    model.load_state_dict(model_paras, strict=False)

    # 冻结特征提取器参数
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = requires_grad

    ema_model_paras = {k.replace('momentum_backbone', 'encoder'): v for k, v in state.items()}
    ema_model_paras = {k.replace('momentum_projector', 'head'): v for k, v in ema_model_paras.items()}
    ema_model_paras = {k.replace('downsample', 'shortcut'): v for k, v in ema_model_paras.items()}
    ema_model.load_state_dict(ema_model_paras, strict=False)

    # 冻结特征提取器参数
    for name, param in ema_model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = requires_grad

    return model, ema_model


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epoch)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().cuda()
        if len(target.shape) == 2:
            target = torch.argmax(target, dim=1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def knn_acc(target_feat, target_label, query_feat=None, query_label=None, only_I=False):
    topK = (1, 2, 3, 4, 5)
    right_num = np.array([0] * 5)
    neighbor_acc = np.array([0] * 5).astype(float)
    target_np = target_feat.cpu().numpy()
    if query_feat is not None and query_label is not None:
        query_np = query_feat.cpu().numpy()
    else:
        query_np = target_np
        query_label = target_label
    index_faiss = faiss.IndexFlatL2(target_np.shape[1])
    index_faiss.add(target_np)
    _, I = index_faiss.search(query_np, k=6)
    if only_I:
        return I
    for ii in range(len(query_np)):
        gt_label = query_label[ii]
        knn_labels = target_label[I[ii, 1:]]
        for item in topK:
            right_num[item - 1] += (knn_labels[0:item] == gt_label).sum()
    for item in topK:
        neighbor_acc[item - 1] = right_num[item - 1] / (item * len(query_np))

    return neighbor_acc


def get_I(model, train_loader_lb, train_loader_ulb):
    model.eval()
    feat_ulb_all = torch.tensor([]).cuda()
    label_ulb_all = torch.tensor([]).cuda()
    for i, (x_weak_ulb, x_strong_ulb, y_ulb, partial_y_ulb, index_ulb) in enumerate(train_loader_ulb):
        x_weak_ulb = x_weak_ulb.cuda()
        y_ulb = y_ulb.cuda()
        outputs = model(x_weak_ulb)

        feat_ulb_all = torch.cat((feat_ulb_all, outputs['feat']))
        label_ulb_all = torch.cat([label_ulb_all, y_ulb])

    feat_lb_all = torch.tensor([]).cuda()
    label_lb_all = torch.tensor([]).cuda()
    for i, (x_weak, x_strong, y, part_y, index) in enumerate(train_loader_lb):
        x_weak = x_weak.cuda()
        y = y.cuda()
        outputs = model(x_weak)

        feat_lb_all = torch.cat((feat_lb_all, outputs['feat']))
        label_lb_all = torch.cat([label_lb_all, y])

    lb_to_lb_I = knn_acc(feat_lb_all, label_lb_all, only_I=True)
    lb_to_ulb_I = knn_acc(feat_lb_all, label_lb_all, feat_ulb_all, label_ulb_all, only_I=True)

    return lb_to_lb_I, lb_to_ulb_I


def disa_acc(train_loader_lb, train_loader_ulb, confidence, confidence_ulb, file_root, epoch):
    disa_acc_lb = (train_loader_lb.dataset.targets.cuda() == torch.max(confidence, dim=1)[1]).sum() / len(confidence)
    disa_acc_ulb = (train_loader_ulb.dataset.targets.cuda() == torch.max(confidence_ulb, dim=1)[1]).sum() / len(
        confidence_ulb)
    with open(file_root + '/transductive_acc.txt', 'a') as f:
        f.write('{} epcoh transductive acc_lb: {:.4f}\t acc_ulb: {:.4f}\n'.format(epoch, disa_acc_lb, disa_acc_ulb))
    # logger.info('{} epcoh transductive acc_lb: {:.4f}\t acc_ulb: {:.4f}'.format(epoch, disa_acc_lb, disa_acc_ulb))




