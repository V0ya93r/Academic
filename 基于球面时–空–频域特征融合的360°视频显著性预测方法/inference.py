import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

import config
from DataLoader360Videos_5 import RGB_and_OF
from utils import frames_extraction
from utils import save_video

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def eval(test_data, model, device, result_imp_path):

    model.to(device)
    model.eval()

    with torch.no_grad():

        for x, names in tqdm.tqdm(test_data):
            # 显式地将数据移到 GPU
            x = x.to(device)  # 将输入数据移到指定的设备（GPU或CPU）
            # 在GPU上进行模型推理
            pred = model(x)  # 进行预测，输入数据已经在 GPU 上

            #pred = model(x.to(device))

            batch_size, Nframes, _,_ = pred[:, :, 0, :, :].shape
            
            for bs in range(batch_size):
                for iFrame in range(4,Nframes):
     
                    folder = os.path.join(result_imp_path, names[iFrame][bs].split('_')[0])
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    sal = pred[bs, iFrame, 0, :, :].detach().cpu().numpy()
                    sal = (sal - sal.min()) / (sal.max() - sal.min())
                    cv2.imwrite(os.path.join(folder, names[iFrame][bs] + '.png'), (sal * 255).astype(np.uint8))


if __name__ == "__main__":

    # 提取视频帧
    if not os.path.exists(os.path.join(config.videos_folder, 'frames')):
        frames_extraction(config.videos_folder)
    print("提取视频帧完成")

    # ‘frames’视频获取名字 Obtain video names from the new folder 'frames'
    inference_frames_folder = os.path.join(config.videos_folder, 'frames')
    video_test_names = os.listdir(inference_frames_folder)
    print("视频切取图片完成")

    # 选择显卡 Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.cuda.current_device()
    print("模型将会在以下GPU运行", torch.cuda.get_device_name(0), "device")

    # 加载模型 Load the model
    model = torch.load(config.inference_model, map_location=device, weights_only=False)
    print("加载模型完成")

    # Load the data. Use the appropiate data loader depending on the expected input data
    if config.of_available:
        test_video360_dataset = RGB_and_OF(inference_frames_folder, config.optical_flow_dir, None, None, video_test_names, config.sequence_length, split='test', load_names=True, skip=0, inference=True)
        #test_video360_dataset = RGB_and_OF(inference_frames_folder, config.optical_flow_dir, None, config.frequency_dir, video_test_names,
        #                               config.sequence_length, split='test', load_names=True, skip=0, inference=True)

    #else:
    #    test_video360_dataset = RGB(inference_frames_folder, None, video_test_names, config.sequence_length, split='test', load_names=True, skip=0, inference=True)

    # DataLoader360Video_3  Load the data. Use the appropiate data loader depending on the expected input data
    #if config.of_available:
    #    test_video360_dataset = RGB_and_OF(inference_frames_folder, config.optical_flow_dir, None, video_test_names,config.sequence_length, split='test', load_names=True, skip=0,inference=True)
    #else:
    #    test_video360_dataset = RGB(inference_frames_folder, None, video_test_names, config.sequence_length,split='test', load_names=True, skip=0, inference=True)
    test_data = DataLoader(test_video360_dataset, batch_size=config.batch_size, shuffle=False)

    eval(test_data, model, device, config.results_dir)

    # Save video with the results

    for video_name in video_test_names:
        save_video(os.path.join(inference_frames_folder, video_name), 
                os.path.join(config.results_dir, video_name),
                None,
                'SST-Sal_pred_' + video_name +'.avi')
