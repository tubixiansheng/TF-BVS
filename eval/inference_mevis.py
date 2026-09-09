import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import torch.nn.functional as F
import json

from tqdm import tqdm
import shutil

import multiprocessing as mp
import threading
import warnings

warnings.filterwarnings("ignore")

from ruamel.yaml import YAML
from easydict import EasyDict
from torch.cuda.amp import autocast
from misc import nested_tensor_from_videos_list

color_list = utils.colormap()
color_list = color_list.astype('uint8').tolist()

transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    print("Inference only supports for batch size = 1")
    print(args)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    save_dir = os.path.join(args.output_dir, args.dataset_name, args.version, split)
    output_dir = os.path.join(save_dir, "Annotations")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src=args.config_path, dst=os.path.join(save_dir, 'config.yaml'))

    save_visualize_path_prefix = os.path.join(save_dir, '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    root = Path(args.dataset_path)
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]

    if args.subset_size > 0:
        new_data = {}
        for video, ref_dict in data.items():
            step = (len(ref_dict['frames']) + args.subset_size - 1) // args.subset_size
            new_frames_list = [[] for _ in range(step)]
            for i, frame in enumerate(ref_dict['frames']):
                new_frames_list[i%step].append(frame)
            for i, frames in enumerate(new_frames_list):
                new_data[video+f'_{i}'] = {
                    'expressions': ref_dict['expressions'],
                    'vid_id': ref_dict['vid_id'],
                    'frames': frames
                }
        data = new_data

    video_list = list(data.keys())

    thread_num = args.num_gpus
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data,
                                                   output_dir, save_visualize_path_prefix,
                                                   img_folder, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" % (total_time))
    print("\nSave results at: {}".format(output_dir))

    if split == "valid":
        print('creating a zip file with the predictions...')
        zip_file_path = os.path.join(save_dir, args.version)
        shutil.make_archive(zip_file_path, 'zip', root_dir=output_dir)
        print('a zip file was successfully created.')
        shutil.rmtree(output_dir)

        print("\nSave results at: {}".format(output_dir))


def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    font_size = 20
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if pid == 0:
        print('number of params:', n_parameters)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint["model_state_dict"]
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    num_all_frames = 0
    model.eval()

    for video in video_list:
        torch.cuda.empty_cache()
        metas = []

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        for i in range(num_expressions):
            meta = {}
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        frames = data[video]["frames"]
        video_name = video.split('_')[0]
        imgs = []
        for t in range(video_len):
            frame = frames[t]
            img_path = os.path.join(img_folder, video_name, frame + ".jpg")
            img = Image.open(img_path).convert('RGB')
            origin_w, origin_h = img.size
            imgs.append(transform(img))
        imgs = torch.stack(imgs, dim=0).to(args.device)
        samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
        target = {"size": size}

        for i in range(num_expressions):
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)

            with torch.no_grad():
                with autocast(args.enable_amp):
                    outputs = model.infer(samples, [exp], [target])

            pred_logits = outputs["pred_logits"][0]
            pred_masks = outputs["pred_masks"][0]
            pred_boxes = outputs["pred_boxes"][0]

            pred_scores = pred_logits.sigmoid()
            pred_scores = pred_scores.mean(0)
            idx = pred_scores.squeeze(-1) > args.obj_threshold
            if args.top1 or not idx.any():
                top_score, idx = pred_scores.squeeze(-1).topk(1, sorted=False)

            pred_masks = pred_masks[:, idx, :, :].cpu()
            pred_boxes = pred_boxes[:, idx, :].cpu()
            pred_logits = pred_logits[:, idx, :].cpu()

            pred_masks = pred_masks[:, :, :img_h, :img_w]
            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False
                                       ) > 0.
            pred_masks = pred_masks.sum(dim=1).clamp(max=1).numpy()
            color = color_list[0]

            if args.visualize:
                for t, frame in enumerate(frames):
                    img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                    source_img = Image.open(img_path).convert('RGBA')

                    draw = ImageDraw.Draw(source_img)
                    for pred_box in pred_boxes[t]:
                        draw_box = pred_box.unsqueeze(0)
                        draw_box = rescale_bboxes(draw_box, (origin_w, origin_h)).tolist()

                        xmin, ymin, xmax, ymax = draw_box[0]
                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color),
                                       width=2)

                    text_position = (10, origin_h - font_size - 10)
                    draw.text(text_position, text, fill=(255, 0, 0, 255), font=font)

                    source_img = vis_add_mask(source_img, pred_masks[t], color_list[i % len(color_list)])

                    save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video_name, str(i))
                    if not os.path.exists(save_visualize_path_dir):
                        os.makedirs(save_visualize_path_dir)
                    save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                    source_img.save(save_visualize_path)
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(video_len):
                frame_name = frames[j]
                mask = pred_masks[j].astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)

        with lock:
            progress.update(1)
    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOS DINO: Inference')
    parser.add_argument('--config_path', '-c', default='configs/mevis_swint.yaml',
                        help='path to configuration file')
    parser.add_argument('--split', required=True,
                        help='valid or valid_u')
    parser.add_argument("--checkpoint_path", '-ckpt', required=True,
                        help="The checkpoint path"
                        )
    parser.add_argument("--version", default="refer_dino",
                        help="the saved ckpt and output version")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument('--num_gpus', '-ng', type=int, required=True,
                        help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obj_threshold", default=0.2, type=float, help="the threshold of object scores")
    parser.add_argument("--tracking_alpha", default=0.1, type=float)
    parser.add_argument("--top1", action='store_true')
    parser.add_argument("--subset_size", default=0, type=int, help="0 indicates use the full video")
    args = parser.parse_args()
    with open(args.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: v['value'] for k, v in config.items()}
    args = {**config, **vars(args)}
    args = EasyDict(args)
    args.GroundingDINO.tracking_alpha = args.tracking_alpha
    main(args)
