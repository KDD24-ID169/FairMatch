import numpy as np
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F


class mymethod(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()

        pretrained = args.dataset == 'cub200'

        self.encoder = base_encoder(feat_dim=args.feat_dim, num_classes=args.num_classes, pretrained=pretrained)

        # momentum encoder
        self.ema_encoder = base_encoder(feat_dim=args.feat_dim, num_classes=args.num_classes, pretrained=pretrained)

        for param_q, param_k in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue_feature", torch.randn(args.moco_queue, args.feat_dim))
        self.queue_feature = F.normalize(self.queue_feature, dim=0)
        self.register_buffer("queue_label", torch.randn(args.moco_queue))
        self.register_buffer("queue_gt_label", torch.randn(args.moco_queue))
        self.register_buffer("queue_max_probs", torch.randn(args.moco_queue))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, img_lb, img_ulb_1=None, img_ulb_2=None, img_ulb_strong=None, partial_Y=None,
                    partial_Y_ulb=None, Y_true=None, Y_true_ulb=None, index_ulb=None, eval_only=False):
        if eval_only:
            logits_lb, feat_lb = self.encoder(img_lb)
            return logits_lb, feat_lb
        else:
            length_lb = len(img_lb)
            inputs = torch.cat((img_lb, img_ulb_1, img_ulb_2))
            logits, feats = self.encoder(inputs)

            logits_lb = logits[:length_lb]
            logits_ulb, logits_ulb_strong = logits[length_lb:].chunk(2)

            feats_ulb, feats_ulb_strong = feats[length_lb:].chunk(2)

            return logits_lb, logits_ulb, feats_ulb, logits_ulb_strong, feats_ulb_strong

    @torch.no_grad()
    def momentum_update_key_encoder(self, args):
        for param_q, param_k in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, labels, max_probs, targets, args):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        try:
            self.queue_feature[ptr:ptr + batch_size, :] = keys
            self.queue_label[ptr:ptr + batch_size] = labels
            self.queue_max_probs[ptr:ptr + batch_size] = max_probs
            self.queue_gt_label[ptr:ptr + batch_size] = targets
            ptr = (ptr + batch_size) % args.moco_queue  # move pointer
        except:
            remain = args.moco_queue - ptr
            temp1 = batch_size - remain
            self.queue_feature[ptr:, :] = keys[:remain, :]
            self.queue_label[ptr:] = labels[:remain]
            self.queue_max_probs[ptr:ptr + batch_size] = max_probs[:remain]
            self.queue_gt_label[ptr:] = targets[:remain]
            ptr = 0
            self.queue_feature[ptr:ptr+temp1, :] = keys[remain:, :]
            self.queue_label[ptr:ptr+temp1] = labels[remain:]
            self.queue_max_probs[ptr:ptr+temp1] = max_probs[remain:]
            self.queue_gt_label[ptr:ptr+temp1] = targets[remain:]

        self.queue_ptr[0] = ptr
