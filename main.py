import argparse
import copy
import random
import dataset as dataset
from utils.utils_algo import over_write_args_from_file, save_checkpoint
from utils.utils_func import *
from utils.model import mymethod
from backbone.resnet import ResNet18

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

def get_config():
    parser = argparse.ArgumentParser(
        description='Revisiting Consistency Regularization for Deep Partial Label Learning')
    # basic paras
    parser.add_argument('-ep', '--epoch', help='number of epochs', type=int, default=200)
    parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=0.1)
    parser.add_argument('-bs', '--batch_size', help='batch size for training', type=int, default=64)
    parser.add_argument('-pr', '--partial_rate', help='partial rate (flipping)', type=float, default=0.3)
    parser.add_argument('-ds', '--dataset', help='specify a dataset', type=str, default='cifar10',
                        choices=['svhn', 'cifar10', 'cifar100', 'fmnist', 'kmnist'])
    parser.add_argument('-nc', '--num_classes', help='num of classes', type=int, default=10)
    parser.add_argument('--backbone', help='backbone name', type=str, default='resnet18', choices=['resnet18'],
                        required=False)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cosine', action='store_true', help='use cosine lr schedule')
    parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument("--seed", help="seed for initializing training.", type=int, default=1)
    parser.add_argument("-gpu", help="GPU id to use.", type=int, default=None)
    parser.add_argument("--hierarchical", type=bool, default=False)

    # semi-supervised settings
    parser.add_argument("--num_labels", help='number of partial samples', type=int, default=250)
    parser.add_argument("--ratio", help='the ratio of unsupervised batch size', type=int, default=7)

    # backbone settings
    parser.add_argument("--feat_dim", help='dimensions of low dimensional feature', type=int, default=256)
    parser.add_argument("--hidden_dim", help='dimensions of hidden layer', type=int, default=2048)

    # Saving & loading of the model
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument("--save_dir", type=str, default="./PLCR_models")
    parser.add_argument("--save_name", "--save_name", type=str, default="PLCR")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)
    parser.add_argument("--use_tensorboard", help="Use tensorboard to plot and save curves", action="store_true")

    # method settings
    parser.add_argument('--lambda_u', type=float, default=1)
    parser.add_argument('--beta_m', type=float, default=1)

    # config file
    parser.add_argument("--c", type=str, default="")

    # add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    save_path = os.path.join(args.save_dir, str(args.seed), args.dataset, str(args.num_labels), args.save_name)
    args.save_path = save_path
    if os.path.exists(save_path) and args.overwrite and args.resume is False:
        import shutil
        shutil.rmtree(save_path)

    if os.path.exists(save_path) and not args.overwrite:
        raise Exception("already existing model: {}".format(save_path))

    args.load_path = os.path.join(args.save_dir, str(args.seed), args.dataset, str(args.num_labels), args.save_name, args.load_name)

    if args.resume:
        if args.load_path is None:
            raise Exception("Resume of training requires --load_path in the args")
        if (
                os.path.abspath(save_path) == os.path.abspath(args.load_path)
                and not args.overwrite
        ):
            raise Exception(
                "Saving & Loading paths are same. \
                            If you want over-write, give --overwrite in the argument."
            )

    # SET save_path, logger and tb_logger
    logger_level = "INFO"
    args.logger = get_logger(args.save_name, args.save_path, logger_level)

    # 不输出到控制台
    args.logger.propagate = False

    args.logger.info(f"Arguments: {args}")
    args.logger.info(f"Use GPU: {args.gpu} for training")

    if args.seed is not None:
        args.logger.info("You have chosen to seed {} training".format(args.seed))
        # random seed has to be set for the synchronization of labeled data sampling in each process.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # gpu setting
    if args.gpu is not None:
        args.logger.info("You have chosen a specific GPU {}.".format(args.gpu))
        torch.cuda.set_device(args.gpu)

    return args


def my_train(args, train_loader_lb, train_loader_ulb, model, optimizer, epoch, criterion, confidence,
             confidence_ulb, loss_cont_fn, label_high, gt_high, acc_high):
    logger = args.logger
    model.train()

    train_loss = AverageMeter()
    supervised_loss = AverageMeter()
    hard_consistent_loss = AverageMeter()
    contrastive_loss = AverageMeter()

    iter_lb = enumerate(train_loader_lb)

    for iter, (x_weak_ulb, x_strong_ulb, y_ulb, partial_y_ulb, index_ulb) in enumerate(train_loader_ulb):
        x_weak_ulb = x_weak_ulb.cuda()
        x_strong_ulb = x_strong_ulb.cuda()
        y_ulb = y_ulb.cuda()
        partial_y_ulb = partial_y_ulb.cuda()

        try:
            _, (x_weak, y, partial_y, index) = next(iter_lb)
        except StopIteration:
            iter_lb = enumerate(train_loader_lb)
            _, (x_weak, y, partial_y, index) = next(iter_lb)

        x_weak = x_weak.cuda()
        y = y.cuda()
        partial_y = partial_y.cuda()

        logits_lb, logits_ulb, feats_ulb, logits_ulb_strong, feats_ulb_strong = model(x_weak, x_weak_ulb, x_strong_ulb,
                                                                                      partial_y, partial_y_ulb, y,
                                                                                      y_ulb, index_ulb)

        probs_lb = F.softmax(logits_lb, dim=-1)
        probs_ulb = F.softmax(logits_ulb, dim=-1)

        if epoch < args.warm_up:
            # supervised loss
            sup_loss = confidence[index] * torch.log(probs_lb+1e-10)
            sup_loss = (-torch.sum(sup_loss)) / sup_loss.size(0)
            supervised_loss.update(sup_loss.item(), len(logits_lb))

            with torch.no_grad():
                model.momentum_update_key_encoder(args)
                _, k = model.ema_encoder(x_strong_ulb)

            add_length = len(feats_ulb)
            features = torch.cat((feats_ulb, k, model.queue_feature.clone().detach()), dim=0)
            max_probs_k, pseudo_labels_k = torch.max(logits_ulb, dim=1)

            model.dequeue_and_enqueue(k, pseudo_labels_k, max_probs_k, y_ulb, args)

            cont_loss = loss_cont_fn(features=features, mask=None, batch_size=add_length)
            contrastive_loss.update(cont_loss.item(), add_length)

            loss = sup_loss + args.loss_weight * cont_loss
            train_loss.update(loss.item())

            if iter % 50 == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t total_loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                            'sup_loss: {sup_loss.val:.4f} ({sup_loss.avg:.4f})\t'
                            'cont_loss: {cont_loss.val:.4f} ({cont_loss.avg:.4f})'.format(epoch, iter,
                                                                                        len(train_loader_ulb),
                                                                                        loss=train_loss,
                                                                                        sup_loss=supervised_loss,
                                                                                        cont_loss=contrastive_loss))
        else:
            sum_high = 0
            for label_idx in range(probs_ulb.size(1)):
                idx_high = probs_ulb[:, label_idx].sort(descending=True)[1][:int(len(probs_ulb) / args.num_classes)]
                sum_high += probs_ulb[idx_high, label_idx].sum().item()
            selected_num = min(
                int(sum_high / len(probs_ulb) * len(partial_y_ulb) / partial_y_ulb.size(1)),
                int(len(partial_y_ulb) / partial_y_ulb.size(1)))
            selected_num = max(selected_num, 1)

            with torch.no_grad():
                pseudo_idx_high = torch.tensor([], dtype=torch.int64).cuda()
                pseudo_label_high = torch.tensor([], dtype=torch.int64).cuda()
                pseudo_max_probs_high = torch.tensor([]).cuda()
                pseudo_gt_high = torch.tensor([], dtype=torch.int64).cuda()
                for label_idx in range(probs_ulb.size(1)):
                    per_label_mask = probs_ulb[:, label_idx].sort(descending=True)[1][:selected_num]
                    pseudo_idx_high = torch.cat((pseudo_idx_high, per_label_mask))
                    pseudo_label_high = torch.cat((pseudo_label_high, torch.tensor([label_idx] * selected_num).cuda()))
                    probs_class = probs_ulb[per_label_mask, label_idx]
                    pseudo_max_probs_high = torch.cat((pseudo_max_probs_high, probs_class))
                    pseudo_gt_high = torch.cat((pseudo_gt_high, y_ulb[per_label_mask]))

            hard_cons_loss = criterion(logits_ulb_strong[pseudo_idx_high, :], pseudo_label_high.long())
            hard_consistent_loss.update(hard_cons_loss.item(), len(pseudo_idx_high))

            # 统计每轮的样本
            if iter == 0:
                sta_label = torch.bincount(pseudo_label_high, minlength=args.num_classes)
                sta_gt = torch.bincount(pseudo_gt_high, minlength=args.num_classes)
                ratio = min(sta_label) / max(sta_label)

                confusion_matrix = torch.zeros((args.num_classes, args.num_classes)).cuda()
                for i in range(len(pseudo_label_high)):
                    confusion_matrix[pseudo_gt_high[i].cpu(), pseudo_label_high[i].cpu()] += 1
                confusion_matrix = confusion_matrix / (confusion_matrix.sum(dim=1) + 1e-10).repeat(confusion_matrix.size(1),
                                                                                                   1).transpose(0, 1)

                label_high.append(sta_label.cpu().tolist())
                gt_high.append(sta_gt.cpu().tolist())
                acc_high.append(torch.diagonal(confusion_matrix).cpu().tolist())

            # Beta分布
            beta_distribution = torch.distributions.Beta(args.alpha_mixup, args.alpha_mixup)
            mixup_weight =  beta_distribution.sample()
            mixup_weight = torch.max(mixup_weight, 1-mixup_weight)
            mixup_idx = torch.randperm(len(pseudo_idx_high))[:len(probs_lb)]

            if len(mixup_idx) >= len(x_weak):
                mixup_x = mixup_weight * x_weak + (1-mixup_weight) * x_weak_ulb[pseudo_idx_high[mixup_idx],:]
                mixup_y = mixup_weight * confidence[index] + (1-mixup_weight) * F.one_hot(pseudo_label_high[mixup_idx], num_classes=args.num_classes)
            else:
                mixup_x = mixup_weight * x_weak[:len(mixup_idx),:] + (1 - mixup_weight) * x_weak_ulb[pseudo_idx_high[mixup_idx], :]
                mixup_y = mixup_weight * confidence[index[:len(mixup_idx)]] + (1-mixup_weight) * F.one_hot(pseudo_label_high[mixup_idx], num_classes=args.num_classes)

            mixup_logits, _ = model.encoder(mixup_x)
            mixup_probs = F.softmax(mixup_logits, dim=-1)

            # supervised loss
            sup_loss_1 = confidence[index] * torch.log(probs_lb+1e-10)
            sup_loss_1 = (-torch.sum(sup_loss_1)) / sup_loss_1.size(0)
            supervised_loss.update(sup_loss_1.item(), len(probs_lb))

            sup_loss_2 = mixup_y * torch.log(mixup_probs+1e-10)
            sup_loss_2 = (-torch.sum(sup_loss_2)) / sup_loss_2.size(0)
            supervised_loss.update(sup_loss_2.item(), len(mixup_probs))

            sup_loss = sup_loss_1 + args.beta_m * sup_loss_2

            loss = sup_loss + args.lambda_u * hard_cons_loss
            train_loss.update(loss.item())

            if iter % 50 == 0:
                confusion_matrix = torch.zeros((args.num_classes, args.num_classes)).cuda()
                for i in range(len(pseudo_label_high)):
                    confusion_matrix[pseudo_gt_high[i].cpu(), pseudo_label_high[i].cpu()] += 1
                confusion_matrix = confusion_matrix / (confusion_matrix.sum(dim=1)+1e-10).repeat(confusion_matrix.size(1), 1).transpose(0, 1)

                select_acc = (pseudo_gt_high==pseudo_label_high).sum() / len(pseudo_label_high)
                with open(args.save_path + '/select_acc.txt', 'a') as f:
                    f.write(
                        '{} epcoh {} iter \t select acc: {:.4f}\t length: {}\n'.format(epoch, iter, select_acc, len(pseudo_label_high)))
                    if args.num_classes == 10:
                        f.write('select confusion matrix:\n{}\n'.format(np.array_str(confusion_matrix.cpu().numpy())))
                    else:
                        f.write('select confusion matrix:\n{}\n'.format(
                            np.array_str(torch.diagonal(confusion_matrix).reshape(10, 10).cpu().numpy())))

                queue_acc = (model.queue_label == model.queue_gt_label).sum() / len(model.queue_label)
                with open(args.save_path + '/queue_acc.txt', 'a') as f:
                    f.write(
                        '{} epcoh {} iter \t queue acc: {:.4f}\n'.format(epoch, iter, queue_acc))

                logger.info('Epoch: [{0}][{1}/{2}]\t total_loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                            'sup_loss: {sup_loss.val:.4f} ({sup_loss.avg:.4f})\t'
                            'hard_cons_loss: {hard_cons_loss.val:.4f} ({hard_cons_loss.avg:.4f})\t'
                            'cont_loss: {cont_loss.val:.4f} ({cont_loss.avg:.4f})'.format(
                    epoch, iter, len(train_loader_ulb), loss=train_loss, sup_loss=supervised_loss,
                    hard_cons_loss=hard_consistent_loss, cont_loss=contrastive_loss))

        with torch.no_grad():
            revisedY = partial_y.clone()
            revisedY[revisedY > 0] = 1
            revisedY = revisedY * probs_lb
            revisedY = revisedY / (revisedY.sum(dim=1) + 1e-10).repeat(revisedY.size(1), 1).transpose(0, 1)
            confidence[index, :] = args.ema_update_confi * confidence[index, :] + (1 - args.ema_update_confi) * revisedY

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    disa_acc(train_loader_lb, train_loader_ulb, confidence, confidence_ulb, args.save_path, epoch)


def my_test(args, test_loader, model, criterion, epoch, find_knn=False):
    losses = AverageMeter()
    top1 = AverageMeter()

    feat_all = torch.tensor([]).cuda()
    label_all = torch.tensor([]).cuda()
    pred_all = torch.tensor([]).cuda()

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()

            pred, feat = model(x, eval_only=True)
            test_loss = criterion(pred, y)

            label_all = torch.cat([label_all, y])

            pred_label = torch.max(pred, dim=1)[1]
            pred_all = torch.cat([pred_all, pred_label])

            if find_knn:
                feat_all = torch.cat((feat_all, feat))

            # measure accuracy and record loss
            prec1 = accuracy(pred.data, y)[0]
            losses.update(test_loss.item(), x.size(0))
            top1.update(prec1.item(), x.size(0))

            if i % 50 == 0:
                args.logger.info('Test: [{0}/{1}]\t'
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(test_loader), loss=losses, top1=top1))

        if find_knn:
            neighbor_acc = knn_acc(feat_all, label_all)
            with open(args.save_path + '/neighbor_acc.txt', 'a') as f:
                f.write('{} epoch neighbor top-1 acc: {:.4f}\t'
                        'top-2 acc: {:.4f}\t top-3 acc: {:.4f}\t top-4 acc: {:.4f}\t top-5 acc: {:.4f}\n'
                        .format(epoch, neighbor_acc[0], neighbor_acc[1], neighbor_acc[2], neighbor_acc[3],
                                neighbor_acc[4]))
    confusion_matrix_test = torch.zeros((args.num_classes, args.num_classes)).cuda()
    label_all = label_all.long()
    pred_all = pred_all.long()
    for i in range(len(label_all)):
        confusion_matrix_test[label_all[i], pred_all[i]] += 1
    confusion_matrix_test = confusion_matrix_test/(confusion_matrix_test.sum(dim=1)+1e-10).repeat(confusion_matrix_test.size(1), 1).transpose(0, 1)

    return top1.avg, losses.avg


def main(args):
    data_dir = os.path.join(args.data_dir, args.dataset.lower())
    logger = args.logger
    save_path = args.save_path

    # load data
    if args.dataset == "cifar10":
        train_loader_lb, train_loader_ulb, test_loader = dataset.cifar10_dataloaders(data_dir, args)
    elif args.dataset == "cifar100":
        train_loader_lb, train_loader_ulb, test_loader = dataset.cifar100_dataloaders(data_dir, args)
    else:
        raise Exception("Missing function to handle this dataset")

    model = mymethod(args, ResNet18).cuda()

    # statistic
    label_high = []
    gt_high = []
    acc_high = []

    # init confidence
    confidence = copy.deepcopy(train_loader_lb.dataset.partial_labels)
    confidence = confidence / confidence.sum(axis=1)[:, None]
    confidence = torch.tensor(confidence).cuda()
    confidence_ulb = copy.deepcopy(train_loader_ulb.dataset.partial_labels)
    confidence_ulb = confidence_ulb / confidence_ulb.sum(axis=1)[:, None]
    confidence_ulb = confidence_ulb.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    loss_cont_fn = SupConLoss(args)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    # start epoch
    start_epoch = 0
    best_acc = 0
    best_epoch = 0

    # distribution
    args.lb_distribution = torch.tensor([1/args.num_classes]*args.num_classes).cuda()
    args.ulb_distribution = torch.tensor([1/args.num_classes]*args.num_classes).cuda()

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            checkpoint = torch.load(args.load_path, map_location="cpu")
            start_epoch = checkpoint["save_epoch"] + 1
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            label_high = checkpoint["label_high"]
            gt_high = checkpoint["gt_high"]
            acc_high = checkpoint["acc_high"]
            confidence = checkpoint["confidence"].cuda()
            best_acc = checkpoint["best_acc"]
            best_epoch = checkpoint["best_epoch"]
            args.lb_distribution = checkpoint["lb_distribution"].cuda()
            args.ulb_distribution = checkpoint["ulb_distribution"].cuda()
            logger.info(f'Resume existing model!')
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))


    logger.info('Model training')
    for epoch in range(start_epoch, args.epoch):
        is_best = False
        my_train(args, train_loader_lb, train_loader_ulb, model, optimizer, epoch, criterion, confidence,
                 confidence_ulb, loss_cont_fn, label_high, gt_high, acc_high)

        adjust_learning_rate(args, optimizer, epoch)

        logger.info('confidence sum: {}'.format(np.array_str(confidence.sum(dim=0).cpu().numpy()).replace('\n', '')))

        valacc, valloss = my_test(args, test_loader, model, criterion, epoch, find_knn=False)

        if valacc > best_acc:
            best_acc = valacc
            is_best = True
            best_epoch = epoch
        logger.info(f'Epoch {epoch} Val Acc: {valacc}% \t Best Val Acc: {best_acc}% on Epoch {best_epoch}\n')
        # save latest and best model
        if epoch != args.warm_up-1:
            best_model_name = '{}/model_best.pth'.format(args.save_path)
        else:
            best_model_name = '{}/model_best_20e.pth'.format(args.save_path)
        save_checkpoint({
            'epoch': epoch + 1,
            'backbone': args.backbone,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'confidence': confidence,
            'label_high': label_high,
            'gt_high': gt_high,
            'acc_high': acc_high,
            'save_epoch': epoch,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'lb_distribution': args.lb_distribution,
            'ulb_distribution': args.ulb_distribution,
        }, is_best=is_best, filename='{}/latest_model.pth'.format(args.save_path),
            best_file_name=best_model_name)

if __name__ == "__main__":
    args = get_config()
    main(args)