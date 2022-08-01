# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""
from indra import api, Loader
import os
import random

import numpy as np
import torch
from PIL import ExifTags, Image
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, letterbox, random_perspective
from utils.general import cv2, xywhn2xyxy, xyxy2xywhn
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def collate_fn(batch):
    im, label, path, shapes = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    return torch.stack(im, 0), torch.cat(label, 0), path, shapes


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      hyp=None,
                      augment=False,
                      rank=-1,
                      workers=8,
                      shuffle=False):
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = api.dataset(path)
        transform = my_transform(transform_sample, imgsz, augment, hyp)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    nw = 0
    # sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = Loader(dataset, num_workers = nw, batch_size = batch_size, shuffle = False, transform_fn = transform, tensors=["images", "boxes", "categories"], collate_fn=collate_fn)
    return loader, dataset


def my_transform(fn, img_size, augment, hyp):
    def inner(sample):
        return fn(sample, img_size, augment, hyp)

    return inner



def resize_im(im, img_size, augment):
    h0, w0, c0 = im.shape  # orig hw
    r = img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (augment or r > 1) else cv2.INTER_AREA
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
    if c0 < 3:
        im = np.pad(im, ((0, 0), (0, 0), (0, 3 - c0)), "constant")
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


def transform_sample(sample, img_size, augment, hyp):
    albumentations = Albumentations() if augment else None
    im = sample["images"]
    # Load image
    img, (h0, w0), (h, w) = resize_im(im, img_size, augment)
    shape = img_size

    img, ratio, pad = letterbox(img, shape, auto=False, scaleup=augment)
    shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

    boxes = sample["boxes"]
    categories = sample["categories"]
    labels = np.concatenate((categories[:, np.newaxis], boxes), axis=1)


    if labels.size:  # normalized xywh to pixel xyxy format
        labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

    if augment:
        img, labels = random_perspective(img,
                                            labels,
                                            degrees=hyp['degrees'],
                                            translate=hyp['translate'],
                                            scale=hyp['scale'],
                                            shear=hyp['shear'],
                                            perspective=hyp['perspective'])

    nl = len(labels)  # number of labels
    if nl:
        labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

    if augment:
        # Albumentations
        img, labels = albumentations(img, labels)
        nl = len(labels)  # update after albumentations

        # HSV color-space
        augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        # Flip up-down
        if random.random() < hyp['flipud']:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if random.random() < hyp['fliplr']:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

    labels_out = torch.zeros((nl, 6))
    if nl:
        labels_out[:, 1:] = torch.from_numpy(labels)

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return torch.from_numpy(img), labels_out, "abc", shapes