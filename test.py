import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine import inference
from model import build_segmenter
from utils.dataset import RefDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--save_pred_dir',
                        default=None,
                        type=str,
                        help='directory for saving predicted binary masks')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    cli_args = parser.parse_args()
    assert cli_args.config is not None
    cfg = config.load_cfg_from_cfg_file(cli_args.config)
    if cli_args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, cli_args.opts)
    cfg.save_pred_dir = cli_args.save_pred_dir
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.exp_name = '_'.join([args.exp_name] + [str(name) for name in [args.ladder_dim, args.nhead, args.dim_ffn, args.multi_stage]])
         
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)
    if args.save_pred_dir:
        args.save_pred_dir = os.path.abspath(args.save_pred_dir)
        os.makedirs(args.save_pred_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build dataset & dataloader
    test_data = RefDataset(lmdb_dir=args.test_lmdb,
                           mask_dir=args.mask_root,
                           dataset=args.dataset,
                           split=args.test_split,
                           mode='test',
                           input_size=args.input_size,
                           word_length=args.word_len)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    args.model_dir = os.path.join(args.output_dir, "best_model.pth")
    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    # inference
    inference(test_loader, model, args)


if __name__ == '__main__':
    main()
