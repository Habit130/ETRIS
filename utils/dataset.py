import json
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s).
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class RefDataset(Dataset):
    def __init__(self,
                 dataset_root,
                 json_path,
                 mode,
                 input_size,
                 word_length,
                 caption_index=2,
                 mask_foreground_threshold=0):
        super(RefDataset, self).__init__()
        self.dataset_root = Path(dataset_root).expanduser()
        if not self.dataset_root.is_absolute():
            self.dataset_root = (Path.cwd() / self.dataset_root).resolve()
        self.json_path = self._resolve_path(json_path)
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.caption_index = int(caption_index)
        self.mask_foreground_threshold = float(mask_foreground_threshold)
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.samples = self._load_samples()

    def _resolve_path(self, path_like):
        path = Path(path_like).expanduser()
        if path.is_absolute():
            return path
        candidate = (self.dataset_root / path).resolve()
        if candidate.exists():
            return candidate
        return (Path.cwd() / path).resolve()

    def _load_samples(self):
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.dataset_root}")
        if not self.json_path.exists():
            raise FileNotFoundError(f"Dataset json does not exist: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        if not isinstance(raw_samples, list):
            raise ValueError(f"Dataset json must be a list: {self.json_path}")

        samples = []
        for idx, sample in enumerate(raw_samples):
            item_id = sample.get("id", str(idx))
            captions = sample.get("caption")
            if not isinstance(captions, list) or len(captions) <= self.caption_index:
                raise IndexError(
                    f"Sample '{item_id}' does not contain caption[{self.caption_index}]"
                )

            image_path = self._resolve_path(sample.get("image", ""))
            mask_path = self._resolve_path(sample.get("mask", ""))
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found for sample '{item_id}': {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for sample '{item_id}': {mask_path}")

            samples.append({
                "id": item_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "caption": captions[self.caption_index],
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        ori_img = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if ori_img is None:
            raise ValueError(f"Failed to read image: {sample['image_path']}")
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]

        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {sample['mask_path']}")
        mask = (mask > self.mask_foreground_threshold).astype(np.float32)

        sent = sample["caption"]
        word_vec = tokenize(sent, self.word_length, True).squeeze(0)

        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        mask = cv2.warpAffine(mask,
                              mat,
                              self.input_size,
                              flags=cv2.INTER_NEAREST,
                              borderValue=0.)
        mask = (mask > 0.5).astype(np.float32)

        if self.mode == "train":
            img, mask = self.convert(img, mask)
            return img, word_vec, mask

        img = self.convert(img)[0]
        params = {
            "item_id": sample["id"],
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
            "caption": sent,
            "inverse": mat_inv.astype(np.float32),
            "ori_size": np.array(img_size, dtype=np.int32),
        }
        return img, word_vec, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"dataset_root={self.dataset_root}, " + \
            f"json_path={self.json_path}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}, " + \
            f"caption_index={self.caption_index}, " + \
            f"mask_foreground_threshold={self.mask_foreground_threshold})"
