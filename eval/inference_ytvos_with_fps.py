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
from PIL import Image, ImageDraw
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


def main(config):
    print("Inference only supports for batch size = 1")
    print(config)

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = "valid"
    save_dir = os.path.join(args.output_dir, args.dataset_name, args.version)
    output_dir = os.path.join(save_dir, "Annotations")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src=args.config_path, dst=os.path.join(save_dir, 'config.yaml'))

    save_visualize_path_prefix = os.path.join(save_dir, split + '_images')
    if config.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    root = Path(args.dataset_path)
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    assert len(video_list) == 202, 'error: incorrect number of validation videos'

    thread_num = config.num_gpus
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, config, data,
                                                   output_dir, save_visualize_path_prefix,
                                                   img_folder, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    total_frames = 0
    total_inference_time = 0.0
    for pid, (frames, inference_time) in result_dict.items():
        total_frames += frames
        total_inference_time += inference_time

    print("\nTotal inference time: %.4f s" % total_inference_time)
    print("Total frames processed: %d" % total_frames)
    print("FPS: %.2f" % (total_frames / total_inference_time))
    print("Total wall clock time: %.4f s" % total_time)

    print('creating a zip file with the predictions...')
    zip_file_path = os.path.join(save_dir, args.version)
    shutil.make_archive(zip_file_path, 'zip', root_dir=save_dir, base_dir='Annotations')
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

    total_frames = 0
    total_inference_time = 0.0

    for video in video_list:
        torch.cuda.empty_cache()
        metas = []

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        frames = data[video]["frames"]
        video_name = video
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
        target = {"size": size, "frames_idx": range(video_len)}

        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                with autocast(args.enable_amp):
                    outputs = model.infer(samples, [exp], [target])

            pred_logits = outputs["pred_logits"][0]
            pred_masks = outputs["pred_masks"][0]
            pred_boxes = outputs["pred_boxes"][0]

            pred_scores = pred_logits.sigmoid()
            pred_scores = pred_scores.mean(0)
            max_scores, _ = pred_scores.max(-1)
            _, max_ind = max_scores.max(-1)
            max_inds = max_ind.repeat(video_len)
            pred_masks = pred_masks[range(video_len), max_inds, ...]
            pred_masks = pred_masks.unsqueeze(0)
            pred_boxes = pred_boxes[range(video_len), max_inds]

            pred_masks = pred_masks[:, :, :img_h, :img_w]
            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
            pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).detach().cpu().numpy()

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            total_inference_time += elapsed_time
            total_frames += video_len

            if args.visualize:
                for t, frame in enumerate(frames):
                    img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                    source_img = Image.open(img_path).convert('RGBA')

                    draw = ImageDraw.Draw(source_img)
                    draw_boxes = pred_boxes[t].unsqueeze(0)
                    draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()

                    xmin, ymin, xmax, ymax = draw_boxes[0]
                    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[i % len(color_list)]),
                                   width=2)

                    source_img = vis_add_mask(source_img, pred_masks[t], color_list[i % len(color_list)])

                    save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
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

    result_dict[str(pid)] = (total_frames, total_inference_time)
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


def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x - 10, y, x + 10, y), tuple(cur_color), width=4)
        draw.line((x, y - 10, x, y + 10), tuple(cur_color), width=4)


def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=tuple(cur_color), outline=tuple(cur_color), width=1)


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
    parser.add_argument('--config_path', '-c', required=True,
                        help='path to configuration file')
    parser.add_argument("--checkpoint_path", '-ckpt', required=True,
                        help="The checkpoint path"
                        )
    parser.add_argument("--version", default="refer_dino",
                        help="the saved ckpt and output version")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument('--num_gpus', '-ng', type=int, required=True,
                        help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with open(args.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: v['value'] for k, v in config.items()}
    args = {**config, **vars(args)}
    args = EasyDict(args)

    main(args)