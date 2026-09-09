import os, json, random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import datasets.transform_video as T
from datasets.categories import ytvos_category_dict as category_dict


def make_coco_transforms(image_set, max_size=256):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.2663, 0.2736, 0.2876], [0.1378, 0.1416, 0.1483])
    ])

    if image_set == "train":
        return T.Compose([T.RandomHorizontalFlip(), normalize])
    if image_set in ["valid", "test"]:
        return T.Compose([normalize])
    raise ValueError(f"unknown split {image_set}")


class ReferYouTubeVOSDataset(Dataset):
    def __init__(self, subset_type, dataset_path, num_frames, **kwargs):
        """Create the text-free training variant of the dataset."""
        self.subset_type  = subset_type
        self.dataset_path = dataset_path
        self.img_folder   = os.path.join(dataset_path, subset_type)
        self.ann_file     = os.path.join(dataset_path, subset_type, "cleaned_output.json")

        self.num_frames   = num_frames
        self._transforms  = make_coco_transforms(subset_type)

        self.textless_prob = float(kwargs.get("textless_prob", 0.0)) if subset_type == "train" else 0.0

        self.metas, self.videos = self.prepare_metas()
        print(f"\n video num: {len(self.videos)}  clip num: {len(self.metas)}\n")

    def prepare_metas(self):
        with open(self.ann_file, "r") as f:
            videos_dict = json.load(f)["videos"]

        if self.subset_type == "train":
            meta_json = os.path.join(self.img_folder, "meta.json")
            with open(meta_json, "r") as f:
                meta_dict = json.load(f)["videos"]

        metas, videos = [], list(videos_dict.keys())
        for vid in videos:
            vinfo = videos_dict[vid]
            exp_dicts = vinfo["expressions"]

            for exp_id, exp_info in exp_dicts.items():
                exp_text = exp_info["exp"]

                if self.subset_type == "train":
                    obj_id = int(exp_info["obj_id"])
                    obj_info = vinfo["objects"][exp_info["obj_id"]]
                    category = obj_info["category"]
                    valid_frames = sorted(obj_info["valid_frames"])

                    if len(valid_frames) == 0:
                        print(f"[Skip] {vid} obj {obj_id} NO valid_frames")
                        continue

                    for start in range(0, len(valid_frames), self.num_frames):
                        metas.append({
                            "video": vid,
                            "exp": exp_text,
                            "obj_id": obj_id,
                            "category": category,
                            "valid_frames": valid_frames,
                            "start_idx": start,
                        })
                else:
                    metas.append({
                        "video": vid,
                        "exp_id": exp_id,
                        "exp": exp_text,
                        "frames": sorted(vinfo["frames"]),
                    })
        return metas, videos

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        if self.subset_type == "train":
            max_retry = 10
            for _ in range(max_retry):
                meta = self.metas[idx]
                video, exp = meta["video"], meta["exp"]
                obj_id, category = meta["obj_id"], meta["category"]
                valid_frames = meta["valid_frames"]
                start_idx = meta["start_idx"]

                if len(valid_frames) < self.num_frames:
                    sample_inds = list(range(len(valid_frames)))
                    extra = random.choices(range(len(valid_frames)), k=self.num_frames - len(sample_inds))
                    sample_inds.extend(extra)
                else:
                    sample_inds = list(range(start_idx, min(start_idx + self.num_frames, len(valid_frames))))
                    if len(sample_inds) < self.num_frames:
                        extra = random.choices(range(len(valid_frames)), k=self.num_frames - len(sample_inds))
                        sample_inds.extend(extra)
                sample_inds = sorted(sample_inds)

                imgs, labels, boxes, masks, valids = [], [], [], [], []
                try:
                    for i in sample_inds:
                        f_name = valid_frames[i]
                        img_path  = os.path.join(self.img_folder, "JPEGImages", video, f"{f_name}.jpg")
                        mask_path = os.path.join(self.img_folder, "Annotations", video, f"{f_name}.png")
                        img  = Image.open(img_path).convert("RGB")
                        mask_np = np.array(Image.open(mask_path).convert("P"))
                        mask = torch.from_numpy((mask_np == 255).astype(np.float32))

                        labels.append(torch.tensor(category_dict[category]))
                        if mask.any():
                            y0, x0 = torch.where(mask)[0].min(), torch.where(mask)[1].min()
                            y1, x1 = torch.where(mask)[0].max(), torch.where(mask)[1].max()
                            boxes.append(torch.tensor([x0, y0, x1, y1], dtype=torch.float))
                            valids.append(1)
                        else:
                            boxes.append(torch.zeros(4, dtype=torch.float))
                            valids.append(0)

                        imgs.append(img)
                        masks.append(mask)
                except FileNotFoundError as e:
                    print("[Warn] File missing:", e)
                    idx = random.randint(0, self.__len__() - 1)
                    continue

                if sum(valids) == 0:
                    idx = random.randint(0, self.__len__() - 1)
                    continue

                w, h = imgs[0].size

                use_text = True
                if self.textless_prob > 0.0 and random.random() < self.textless_prob:
                    use_text = False

                caption_text = " ".join(exp.lower().split()) if use_text else None

                target = {
                    "frames_idx": torch.tensor(sample_inds),
                    "labels":     torch.stack(labels),
                    "boxes":      torch.stack(boxes),
                    "masks":      torch.stack(masks),
                    "valid":      torch.tensor(valids),
                    "video":      video,
                    "caption":    caption_text,
                    "has_text":   int(use_text),
                    "orig_size":  torch.as_tensor([h, w]),
                    "size":       torch.as_tensor([h, w]),
                }
                imgs, target = self._transforms(imgs, target)
                return torch.stack(imgs), target, video

            raise RuntimeError(f"No valid sample after {max_retry} attempts for idx {idx}")

        meta = self.metas[idx]
        video, frames, exp = meta["video"], meta["frames"], meta["exp"]
        img_dir = os.path.join(self.img_folder, "JPEGImages", video)
        imgs = [Image.open(os.path.join(img_dir, f"{f}.jpg")).convert("RGB") for f in frames]
        w, h = imgs[0].size

        target = {
            "video_id": video,
            "exp_id":   meta["exp_id"],
            "frame_indices": frames,
            "caption":  " ".join(exp.lower().split()),
            "orig_size": (h, w),
            "size": torch.as_tensor([h, w]),
        }
        imgs, target = self._transforms(imgs, target)
        return torch.stack(imgs), target
