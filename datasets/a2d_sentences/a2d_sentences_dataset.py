import json
import torch
import numpy as np
from torchvision.io import read_video
import h5py
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from os import path
import datasets.transform_video as T
from pycocotools.mask import encode, area
import random


def get_image_id(video_id, frame_idx, ref_instance_a2d_id):
    image_id = f'v_{video_id}_f_{frame_idx}_i_{ref_instance_a2d_id}'
    return image_id

class A2DSentencesDataset(Dataset):
    """
    A Torch dataset for A2D-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """

    def __init__(self, subset_type: str = 'train', dataset_path: str = 'data/a2d_sentences', **kwargs):
        super(A2DSentencesDataset, self).__init__()
        assert subset_type in ['train', 'test'], 'error, unsupported dataset subset type. supported: train, test'
        self.subset_type = subset_type
        self.mask_annotations_dir = path.join(dataset_path, 'text_annotations/a2d_annotation_with_instances')
        self.videos_dir = path.join(dataset_path, 'Release/clips320H')
        self.ann_file = path.join(dataset_path, f'a2d_sentences_single_frame_{subset_type}_annotations.json')
        self.text_annotations = self.get_text_annotations()

        self._transforms = make_coco_transforms(subset_type)
        self.num_frames = 1

        print(f'\n{subset_type} sample num: ', len(self.text_annotations))
        print('\n')

    def get_text_annotations(self):
        with open(str(self.ann_file), 'r') as f:
            text_annotations_by_frame = [tuple(a) for a in json.load(f)]
            return text_annotations_by_frame

    @staticmethod
    def bounding_box(img: torch.tensor):
        rows = torch.any(img, dim=1)
        cols = torch.any(img, dim=0)
        y0, y1 = torch.where(rows)[0][[0, -1]]
        x0, x1 = torch.where(cols)[0][[0, -1]]
        return x0, y0, x1, y1

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            text_query, video_id, frame_idx, instance_id = self.text_annotations[idx]

            text_query = " ".join(text_query.lower().split())

            video_frames, _, _ = read_video(path.join(self.videos_dir, f'{video_id}.mp4'), pts_unit='sec')
            vid_len = len(video_frames)
            frame_id = frame_idx - 1

            sample_indx = [frame_id]

            if self.num_frames != 1:
                if self.subset_type == 'train':
                    sample_id_before = random.randint(1, 3)
                    sample_id_after = random.randint(1, 3)
                    local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                    sample_indx.extend(local_indx)

                    if self.num_frames > 3:
                        all_inds = list(range(vid_len))
                        global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                        global_n = self.num_frames - len(sample_indx)
                        if len(global_inds) > global_n:
                            select_id = random.sample(range(len(global_inds)), global_n)
                            for s_id in select_id:
                                sample_indx.append(global_inds[s_id])
                        elif vid_len >= global_n:
                            select_id = random.sample(range(vid_len), global_n)
                            for s_id in select_id:
                                sample_indx.append(all_inds[s_id])
                        else:
                            select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                            for s_id in select_id:
                                sample_indx.append(all_inds[s_id])
                else:
                    start_idx, end_idx = frame_id - self.num_frames // 2, frame_id + (self.num_frames + 1) // 2
                    sample_indx = []
                    for i in range(start_idx, end_idx):
                        i = min(max(i, 0), len(video_frames) - 1)
                        sample_indx.append(i)

                sample_indx.sort()
            valid_indices = sample_indx.index(frame_id)

            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                img = F.to_pil_image(video_frames[frame_indx].permute(2, 0, 1))
                imgs.append(img)

            frame_annot_path = path.join(self.mask_annotations_dir, video_id, f'{frame_idx:05d}.h5')
            f = h5py.File(frame_annot_path)
            instances = list(f['instance'])
            instance_idx = instances.index(instance_id)

            instance_masks = np.array(f['reMask'])
            if len(instances) == 1:
                instance_masks = instance_masks[np.newaxis, ...]
            instance_masks = torch.tensor(instance_masks).transpose(1, 2)
            mask_rles = [encode(mask) for mask in instance_masks.numpy()]
            mask_areas = area(mask_rles).astype(np.float64)
            f.close()

            label = torch.tensor(0, dtype=torch.long)
            mask = instance_masks[instance_idx]
            if mask.any():
                box = torch.tensor(self.bounding_box(mask), dtype=torch.float)
                valid.append(1)
            else:
                box = torch.zeros(4, dtype=torch.float)
                valid.append(0)

            labels.append(label)
            boxes.append(box)
            masks.append(mask)

            h, w = instance_masks.shape[-2:]
            labels = torch.stack(labels, dim=0)
            boxes = torch.stack(boxes, dim=0)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)
            target = {
                'frames_idx': torch.tensor(sample_indx),
                'valid_indices': torch.tensor([valid_indices]),
                'labels': labels,
                'boxes': boxes,
                'masks': masks,
                'valid': torch.tensor(valid),
                'caption': text_query,
                'orig_size': (int(h), int(w)),
                'size': torch.as_tensor([int(h), int(w)]),
                'image_id': get_image_id(video_id, frame_idx, instance_id)
            }

            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)

            if target['valid'].any():
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target

    def __len__(self):
        return len(self.text_annotations)


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
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
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


if __name__ == '__main__':
    dataset = A2DSentencesDataset()
