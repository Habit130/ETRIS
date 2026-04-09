import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


def _get_pred_threshold(args):
    return getattr(args, 'pred_threshold', 0.35)


def _accumulate_binary_stats(pred, mask):
    pred = pred.astype(bool)
    mask = mask.astype(bool)
    tp = np.logical_and(pred, mask).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(mask)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), mask).sum(dtype=np.float64)
    tn = np.logical_and(np.logical_not(pred), np.logical_not(mask)).sum(
        dtype=np.float64)
    return tp, fp, fn, tn


def _compute_binary_metrics(tp, fp, fn, tn):
    eps = 1e-6
    fg_iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    bg_iou = tn / (tn + fp + fn + eps)
    bg_acc = tn / (tn + fp + eps)
    return {
        'IoU': fg_iou,
        'Dice': dice,
        'Recall': recall,
        'mIoU': (fg_iou + bg_iou) / 2.0,
        'mACC': (recall + bg_acc) / 2.0,
    }


def _format_metric_log(metrics):
    return '  '.join(
        f'{name}={100.0 * value:.2f}' for name, value in metrics.items())


def _mask_to_gray(mask):
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[2] == 3:
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.ndim == 3 and mask.shape[2] == 4:
        return cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f'Unsupported mask shape: {mask.shape}')


def _match_mask_layout(pred_mask, ref_mask):
    pred_mask = pred_mask.astype(ref_mask.dtype, copy=False)
    if ref_mask.ndim == 2:
        return pred_mask
    return np.repeat(pred_mask[:, :, None], ref_mask.shape[2], axis=2)


def _save_pred_mask(args, seg_id, sent_idx, pred_mask, ref_mask):
    if not getattr(args, 'save_pred_masks', False):
        return
    filename = f'{seg_id}.png' if sent_idx == 0 else f'{seg_id}-{sent_idx:02d}.png'
    output = _match_mask_layout(pred_mask, ref_mask)
    cv2.imwrite(os.path.join(args.pred_mask_dir, filename), output)


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)

        # forward
        with amp.autocast():
            pred, target, loss = model(image, text, target)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(pred, target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0]:
                wandb.log(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss": loss_meter.val,
                        "training/iou": iou_meter.val,
                        "training/prec@50": pr_meter.val,
                    },
                    step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate(val_loader, model, epoch, args):
    stats = None
    threshold = _get_pred_threshold(args)
    model.eval()
    time.sleep(2)
    for imgs, texts, param in val_loader:
        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        if stats is None:
            stats = torch.zeros(4, dtype=torch.float64, device=imgs.device)
        # inference
        preds = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                  size=imgs.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        # process one batch
        for pred, mask_dir, mat, ori_size in zip(preds, param['mask_dir'],
                                                 param['inverse'],
                                                 param['ori_size']):
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > threshold)
            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            stats += torch.tensor(_accumulate_binary_stats(pred, mask),
                                  dtype=torch.float64,
                                  device=imgs.device)
    if stats is None:
        stats = torch.zeros(4,
                            dtype=torch.float64,
                            device=next(model.parameters()).device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    metrics = _compute_binary_metrics(*stats.tolist())
    logger.info('Evaluation: Epoch=[{}/{}]  {}'.format(
        epoch, args.epochs, _format_metric_log(metrics)))
    return metrics['IoU'], metrics


@torch.no_grad()
def inference(test_loader, model, args):
    threshold = _get_pred_threshold(args)
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        ref_mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_UNCHANGED)
        mask_gray = _mask_to_gray(ref_mask)
        mask_binary = mask_gray / 255.0
        # dump image & mask
        if args.visualize:
            seg_id = param['seg_id'][0]
            if torch.is_tensor(seg_id):
                seg_id = seg_id.item()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
                        img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
                        img=mask_gray)
        else:
            seg_id = param['seg_id'][0]
            if torch.is_tensor(seg_id):
                seg_id = seg_id.item()
        # multiple sentences
        for sent_idx, sent in enumerate(param['sents']):
            text = tokenize(sent, args.word_len, True)
            text = text.cuda(non_blocking=True)
            # inference
            pred = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True).squeeze()
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > threshold)
            tp, fp, fn, tn = _accumulate_binary_stats(pred, mask_binary)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            iou = tp / (tp + fp + fn + 1e-6)
            pred_mask = np.array(pred * 255, dtype=np.uint8)
            _save_pred_mask(args, seg_id, sent_idx, pred_mask, ref_mask)
            # dump prediction
            if args.visualize:
                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou*100, sent)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name),
                            img=pred_mask)
    logger.info('=> Metric Calculation <=')
    metrics = _compute_binary_metrics(total_tp, total_fp, total_fn, total_tn)
    logger.info(_format_metric_log(metrics))
    return metrics['IoU'], metrics
