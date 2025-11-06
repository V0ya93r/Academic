import torch
import os
import cv2
import math
from torch.utils.data import Dataset
import numpy as np
import time
from tqdm import tqdm


class RGB_and_OF(Dataset):
    def __init__(self, path_to_frames, path_to_flow_maps, path_to_saliency_maps, path_to_frequency_maps,
                 video_names, frames_per_data=20, split_percentage=0.2, split='train',
                 resolution=[240, 320], skip=20, load_names=False, transform=False, inference=False):
        self.sequences = []
        self.frames_per_data = frames_per_data
        self.path_frames = path_to_frames
        self.flow_maps = path_to_flow_maps
        self.path_sal_maps = path_to_saliency_maps
        self.path_freq_maps = path_to_frequency_maps
        self.resolution = resolution
        self.load_names = load_names
        self.transform = transform
        self.inference = inference

        # split_percentage方法把video_names划分为训练集、验证集
        sp = int(math.ceil(split_percentage * len(video_names)))
        if split == "validation":
            video_names = video_names[:sp]  # 前sp个视频验证集
        elif split == "train":
            video_names = video_names[sp:]  # 剩余视频训练集

        for name in tqdm(video_names, desc=f"处理{split}集视频"):
            video_folder = os.path.join(self.path_frames, name)

            # 检查视频文件夹是否存在
            if not os.path.exists(video_folder):
                print(f"[警告] 未找到帧文件夹: {video_folder}, 跳过。")
                continue

            # 获取并排序帧文件
            try:
                video_frames_names = sorted(
                    [f for f in os.listdir(video_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
                )
            except (ValueError, IndexError) as e:
                print(f"[警告] 无法解析视频 {name} 的帧文件名: {e}, 跳过。")
                continue

            if len(video_frames_names) == 0:
                print(f"[警告] 视频 {name} 中没有找到帧文件, 跳过。")
                continue

            # 跳过前 skip 帧
            sts = skip

            # 确定推理模式下的帧数
            current_frames_per_data = self.frames_per_data - 4 if inference else self.frames_per_data

            # 将视频分割成等长的序列
            initial_frame = self.frames_per_data + skip

            # 确保不超过视频长度
            if initial_frame >= len(video_frames_names):
                print(f"[警告] 视频 {name} 太短 ({len(video_frames_names)} 帧), 需要的帧数: {initial_frame}, 跳过。")
                continue

            for end in range(initial_frame, len(video_frames_names), current_frames_per_data):
                # 检查序列中所有帧是否存在对应的光流、显著性和频率图
                valid_sequence = True
                sequence_frames = video_frames_names[sts:end]

                for frame_name in sequence_frames:
                    # 检查光流图是否存在
                    flow_path = os.path.join(self.flow_maps, name, frame_name)
                    if not os.path.exists(flow_path):
                        print(f"[警告] 未找到光流图: {flow_path}")
                        valid_sequence = False
                        break

                    # 检查显著性图是否存在（如果提供了路径）
                    if self.path_sal_maps is not None:
                        sal_path = os.path.join(self.path_sal_maps, name, frame_name)
                        if not os.path.exists(sal_path):
                            print(f"[警告] 未找到显著性图: {sal_path}")
                            valid_sequence = False
                            break

                    # 检查频率图是否存在（如果提供了路径）
                    if self.path_freq_maps is not None:
                        freq_path = os.path.join(self.path_freq_maps, name, frame_name)
                        if not os.path.exists(freq_path):
                            print(f"[警告] 未找到频率图: {freq_path}")
                            valid_sequence = False
                            break

                if valid_sequence:
                    self.sequences.append({
                        'video_name': name,
                        'frame_names': sequence_frames
                    })

                sts = end
                if inference:
                    sts = sts - 4  # 推理时重叠序列以获得平滑预测（4帧）

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_img = []
        label = []
        flow_map = []
        freq_map = []
        frame_names = []

        # 获取序列信息
        sequence_info = self.sequences[idx]
        video_name = sequence_info['video_name']
        sequence_frame_names = sequence_info['frame_names']

        # 读取序列中每一帧的RGB图像、光流、显著性和频率图
        for frame_name in sequence_frame_names:
            # 获取帧名（不含扩展名）
            fn = os.path.splitext(frame_name)[0]
            frame_names.append(fn)

            # 读取RGB帧
            frame_path = os.path.join(self.path_frames, video_name, frame_name)
            img_frame = cv2.imread(frame_path)
            if img_frame is None:
                raise ValueError(f'无法读取图像帧: {frame_path}')

            if img_frame.shape[1] != self.resolution[1] or img_frame.shape[0] != self.resolution[0]:
                img_frame = cv2.resize(img_frame, (self.resolution[1], self.resolution[0]),
                                       interpolation=cv2.INTER_AREA)
            img_frame = img_frame.astype(np.float32) / 255.0
            img_frame = torch.FloatTensor(img_frame).permute(2, 0, 1)
            frame_img.append(img_frame.unsqueeze(0))

            # 读取显著性图（如果提供了路径）
            if self.path_sal_maps is not None:
                sal_map_path = os.path.join(self.path_sal_maps, video_name, frame_name)
                saliency_img = cv2.imread(sal_map_path, cv2.IMREAD_GRAYSCALE)

                if saliency_img is None:
                    raise ValueError(f'无法读取显著性图: {sal_map_path}')

                if saliency_img.shape[1] != self.resolution[1] or saliency_img.shape[0] != self.resolution[0]:
                    saliency_img = cv2.resize(saliency_img, (self.resolution[1], self.resolution[0]),
                                              interpolation=cv2.INTER_AREA)
                saliency_img = saliency_img.astype(np.float32)
                # 归一化到0-1范围
                if saliency_img.max() > 0:
                    saliency_img = saliency_img / saliency_img.max()
                saliency_img = torch.FloatTensor(saliency_img).unsqueeze(0)
                label.append(saliency_img.unsqueeze(0))

            # 读取光流图
            flow_map_path = os.path.join(self.flow_maps, video_name, frame_name)
            flow_img = cv2.imread(flow_map_path)

            if flow_img is None:
                raise ValueError(f'无法读取光流图: {flow_map_path}')

            if flow_img.shape[1] != self.resolution[1] or flow_img.shape[0] != self.resolution[0]:
                flow_img = cv2.resize(flow_img, (self.resolution[1], self.resolution[0]),
                                      interpolation=cv2.INTER_AREA)
            flow_img = flow_img.astype(np.float32) / 255.0
            flow_img = torch.FloatTensor(flow_img).permute(2, 0, 1)
            flow_map.append(flow_img.unsqueeze(0))

            # 读取频率图（如果提供了路径）
            if self.path_freq_maps is not None:
                freq_map_path = os.path.join(self.path_freq_maps, video_name, frame_name)
                freq_img = cv2.imread(freq_map_path, cv2.IMREAD_GRAYSCALE)

                if freq_img is None:
                    raise ValueError(f'无法读取频率图: {freq_map_path}')

                if freq_img.shape[1] != self.resolution[1] or freq_img.shape[0] != self.resolution[0]:
                    freq_img = cv2.resize(freq_img, (self.resolution[1], self.resolution[0]),
                                          interpolation=cv2.INTER_AREA)
                freq_img = freq_img.astype(np.float32)
                # 归一化到0-1范围
                if freq_img.max() > 0:
                    freq_img = freq_img / freq_img.max()
                freq_img = torch.FloatTensor(freq_img).unsqueeze(0)
                freq_map.append(freq_img.unsqueeze(0))

        # 准备样本 - 合并RGB、光流和频率数据
        rgb_frames = torch.cat(frame_img, 0)
        flow_frames = torch.cat(flow_map, 0)

        # 合并RGB和光流数据
        if self.path_freq_maps is not None:
            freq_frames = torch.cat(freq_map, 0)
            # 将频率图复制到3个通道以匹配RGB和光流的维度
            freq_frames = freq_frames.repeat(1, 3, 1, 1)
            combined_data = torch.cat((rgb_frames, flow_frames, freq_frames), dim=1)
        else:
            combined_data = torch.cat((rgb_frames, flow_frames), dim=1)

        if self.load_names:
            if self.path_sal_maps is None:
                sample = [combined_data, frame_names]
            else:
                saliency_labels = torch.cat(label, 0)
                sample = [combined_data, saliency_labels, frame_names]
        else:
            if self.path_sal_maps is None:
                sample = [combined_data]
            else:
                saliency_labels = torch.cat(label, 0)
                sample = [combined_data, saliency_labels]

        if self.transform:
            # 假设Rotate类已定义
            tf = Rotate()
            return tf(sample)

        return sample


class Rotate(object):
    """
    它接收一个样本（包含输入和显著性图），然后随机选择一个移位位置t，将输入和显著性图在宽度维度上循环移位（t:end和0:t拼接）
    Rotate the 360º image with respect to the vertical axis on the sphere.
    """

    def __call__(self, sample):
        input = sample[0]
        sal_map = sample[1]

        t = np.random.randint(input.shape[-1])

        new_sample = sample
        new_sample[0] = torch.cat((input[:, :, :, t:], input[:, :, :, 0:t]), dim=3)
        new_sample[1] = torch.cat((sal_map[:, :, :, t:], sal_map[:, :, :, 0:t]), dim=3)

        return new_sample