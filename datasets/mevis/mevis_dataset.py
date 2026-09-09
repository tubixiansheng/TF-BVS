"""
MeViS data loader
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
import datasets.transform_video as T

import os
from PIL import Image
import json
import numpy as np
import random


from pycocotools import mask as coco_mask


class MeViSDataset(Dataset):
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    def __init__(self, subset_type='train', dataset_path='data/mevis', num_frames=16, sampling_step=8,
                 **kwargs):
        if subset_type == 'test':
            subset_type = 'valid_u'
        self.subset_type = subset_type

        self.img_folder = os.path.join(dataset_path, subset_type)
        self.ann_file = os.path.join(dataset_path, subset_type, 'meta_expressions.json')

        self.num_frames = num_frames
        self.sampling_step = sampling_step
        self.prepare_metas()
        self._transforms = make_coco_transforms(subset_type)

        if subset_type == 'train':
            self.mask_dict = json.load(open(os.path.join(dataset_path, subset_type, 'mask_dict.json')))
        print('video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.sampling_step):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                    meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['category'] = 0
                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]

            video, exp, anno_id, category, frames, frame_id = \
                meta['video'], meta['exp'], meta['anno_id'], meta['category'], meta['frames'], meta['frame_id']
            exp = " ".join(exp.lower().split())
            vid_len = len(frames)

            num_frames = self.num_frames
            sample_indx = [frame_id]
            if self.num_frames != 1:
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >= global_n:
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = np.random.choice(range(vid_len), global_n - vid_len).tolist() + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])

            sample_indx.sort()
            if self.subset_type == "train" and np.random.rand() < 0.3:
                sample_indx = sample_indx[::-1]

            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                img = Image.open(img_path).convert('RGB')
                mask = np.zeros(img.size[::-1], dtype=np.float32)
                for x in anno_id:
                    frm_anno = self.mask_dict[x][frame_indx]
                    if frm_anno is not None:
                        mask += coco_mask.decode(frm_anno)

                label = torch.tensor(category)

                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else:
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)
                mask = torch.from_numpy(mask)

                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            w, h = img.size
            labels = torch.stack(labels, dim=0)
            boxes = torch.stack(boxes, dim=0)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)
            target = {
                'frames_idx': torch.tensor(sample_indx),
                'labels': labels,
                'boxes': boxes,
                'masks': masks,
                'valid': torch.tensor(valid),
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]),
                'size': torch.as_tensor([int(h), int(w)])
            }

            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)

            if torch.any(target['valid'] == 1):
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])

    if image_set == 'valid_u':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')