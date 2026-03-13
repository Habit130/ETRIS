import os
import time

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from loguru import logger
from utils.misc import AverageMeter, ProgressMeter, trainMetricGPU


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
def inference(test_loader, model, args):
    def safe_div(num, den):
        return float(num) / float(den) if den else 0.0

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for img, text, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_path'][0], flags=cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {param['mask_path'][0]}")
        mask = (mask > args.mask_foreground_threshold).astype(np.uint8)
        # dump image & mask
        if args.visualize:
            item_id = param['item_id'][0]
            image = cv2.imread(param['image_path'][0], flags=cv2.IMREAD_COLOR)
            if image is not None:
                cv2.imwrite(filename=os.path.join(args.vis_dir, f'{item_id}-img.jpg'),
                            img=image)
            cv2.imwrite(filename=os.path.join(args.vis_dir, f'{item_id}-mask.png'),
                        img=np.array(mask * 255, dtype=np.uint8))

        pred = model(img, text)
        pred = torch.sigmoid(pred)
        if pred.shape[-2:] != img.shape[-2:]:
            pred = F.interpolate(pred,
                                 size=img.shape[-2:],
                                 mode='bicubic',
                                 align_corners=True)
        pred = pred.squeeze().cpu().numpy()

        h, w = param['ori_size'].numpy()[0]
        mat = param['inverse'].numpy()[0]
        pred = cv2.warpAffine(pred, mat, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderValue=0.)
        pred = np.array(pred > args.pred_threshold, dtype=np.uint8)

        tp += int(np.logical_and(pred == 1, mask == 1).sum())
        fp += int(np.logical_and(pred == 1, mask == 0).sum())
        fn += int(np.logical_and(pred == 0, mask == 1).sum())
        tn += int(np.logical_and(pred == 0, mask == 0).sum())

        if args.visualize:
            item_id = param['item_id'][0]
            pred_name = f"{item_id}-pred.png"
            cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name),
                        img=np.array(pred * 255, dtype=np.uint8))

    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    recall = safe_div(tp, tp + fn)
    iou_bg = safe_div(tn, tn + fn + fp)
    acc_fg = safe_div(tp, tp + fn)
    acc_bg = safe_div(tn, tn + fp)
    miou = (iou + iou_bg) / 2.0
    macc = (acc_fg + acc_bg) / 2.0

    metrics = {
        "iou": iou,
        "dice": dice,
        "recall": recall,
        "miou": miou,
        "macc": macc,
    }
    for key, value in metrics.items():
        logger.info('{}={:.4f}'.format(key, value))

    return metrics
