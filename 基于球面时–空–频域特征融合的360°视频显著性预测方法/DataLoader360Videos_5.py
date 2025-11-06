import torch
import os
import cv2
import math
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class RGB_and_OF(Dataset):
    """
    仅包含 RGB (3通道) + 光流 (3通道) = 共6通道
    用于仅使用帧与光流进行推理
    """

    def __init__(self, path_to_frames, path_to_flow_maps,
                 video_names, frames_per_data=20, split_percentage=0.2, split='train',
                 resolution=[240, 320], skip=20, load_names=False, transform=False, inference=False):
        self.sequences = []
        self.frames_per_data = frames_per_data
        self.path_frames = path_to_frames
        self.flow_maps = path_to_flow_maps
        self.resolution = resolution
        self.load_names = load_names
        self.transform = transform
        self.inference = inference

        # 数据划分
        sp = int(math.ceil(split_percentage * len(video_names)))
        if split == "validation":
            video_names = video_names[:sp]
        elif split == "train":
            video_names = video_names[sp:]

        for name in tqdm(video_names, desc=f"处理{split}集视频"):
            video_folder = os.path.join(self.path_frames, name)
            if not os.path.exists(video_folder):
                print(f"[警告] 未找到帧文件夹: {video_folder}, 跳过。")
                continue

            frame_files = sorted(
                [f for f in os.listdir(video_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
                key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
            )
            if len(frame_files) == 0:
                continue

            # 推理模式下帧数略减
            current_len = self.frames_per_data - 4 if inference else self.frames_per_data
            initial_frame = self.frames_per_data + skip
            if initial_frame >= len(frame_files):
                continue

            for end in range(initial_frame, len(frame_files), current_len):
                start = end - self.frames_per_data
                if start < 0:
                    start = 0
                seq = frame_files[start:end]
                if len(seq) == self.frames_per_data:
                    self.sequences.append({'video_name': name, 'frame_names': seq})

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        video_name = seq['video_name']
        sequence_frame_names = seq['frame_names']

        frame_names = []
        rgb_list, flow_list, gray_uint8_list = [], [], []

        for frame_name in sequence_frame_names:
            frame_names.append(os.path.splitext(frame_name)[0])
            # --- RGB ---
            frame_path = os.path.join(self.path_frames, video_name, frame_name)
            img_bgr = cv2.imread(frame_path)
            if img_bgr is None:
                raise ValueError(f"无法读取图像帧: {frame_path}")
            if img_bgr.shape[1] != self.resolution[1] or img_bgr.shape[0] != self.resolution[0]:
                img_bgr = cv2.resize(img_bgr, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            gray_uint8_list.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
            img_tensor = torch.from_numpy((img_bgr.astype(np.float32) / 255.0)).permute(2, 0, 1)
            rgb_list.append(img_tensor.unsqueeze(0))

            # --- Optical Flow ---
            flow_path = os.path.join(self.flow_maps, video_name, frame_name)
            flow_img = cv2.imread(flow_path)
            if flow_img is None:
                raise ValueError(f"无法读取光流图: {flow_path}")
            if flow_img.shape[1] != self.resolution[1] or flow_img.shape[0] != self.resolution[0]:
                flow_img = cv2.resize(flow_img, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            flow_tensor = torch.from_numpy((flow_img.astype(np.float32) / 255.0)).permute(2, 0, 1)
            flow_list.append(flow_tensor.unsqueeze(0))

        rgb_frames = torch.cat(rgb_list, dim=0)   # (T,3,H,W)
        flow_frames = torch.cat(flow_list, dim=0) # (T,3,H,W)

        combined = torch.cat((rgb_frames, flow_frames), dim=1)  # (T,6,H,W)

        padding = torch.zeros((combined.shape[0], 3, combined.shape[2], combined.shape[3]))
        combined = torch.cat((combined, padding), dim=1)  # (T,9,H,W)

        return combined, frame_names
