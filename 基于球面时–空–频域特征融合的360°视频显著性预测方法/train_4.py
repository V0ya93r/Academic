import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from DataLoader360Video_4 import RGB_and_OF
from sphericalKLDiv import  KLWeightedLossSequence
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import time
from torch.utils.data import DataLoader
import models
from utils import read_txt_file
import config

def train(train_data, val_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='Model'):#默认参数没有传递main

    #日志和保存路径
    writer = SummaryWriter(os.path.join(config.runs_data_dir, model_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'))
    path = os.path.join(config.models_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckp_path = os.path.join(config.ckp_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    os.mkdir(path)
    os.mkdir(ckp_path)

    #优化器和设备设置。SGD随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    criterion.cuda(device)

    print("Training model ...")
    epoch_times = []



    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss_train = 0.
        avg_loss_val = 0.
        counter_train = 0
        counter_val = 0
            
        for x, y in train_data:
            model.zero_grad()#清除模型梯度避免梯度累计
            pred = model(x.to(device))#数据移动gpu，向前传播
            loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))#损失计算+反向传播更新模型参数
            loss.sum().backward()
            optimizer.step()

            avg_loss_train += loss.sum().item()

            counter_train += 1
            if counter_train % 20 == 0:#epoch当前训练轮次   counter_train当前训练批次batch的计数器  len训练数据集的批次数，最后一个平均损失
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter_train, len(train_data),
                                                                                        avg_loss_train / counter_train))

        current_time = time.time()
        print("Epoch {}/{} , Total Spherical KLDiv Loss: {}".format(epoch, EPOCHS, avg_loss_train / counter_train))
        print("Total Time: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        # Evaluate on validation set验证集
        with torch.no_grad():
            for x, y in val_data:
                counter_val += 1
                pred = model(x.to(device))
                loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))
                avg_loss_val += loss.sum().item()

        writer.add_scalars('Loss', {'train': avg_loss_train / counter_train, 'val': avg_loss_val / counter_val}, epoch)
        
        # Save checkpoint and model every 10 epochs
        if epoch % 10 == 0:
            torch.save(model, path + '/'+ str(epoch)+ '_model.pth')
            ckp_path = os.path.join(config.ckp_dir,model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
            os.mkdir(ckp_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')
    
    # Save final model and checkpoints
    torch.save(model, path + '/model.pth')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')

    return model


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Train SST-Sal
    #模型和数据加载
    model = models.SST_Sal(hidden_dim=config.hidden_dim)
    #损失函数
    criterion = KLWeightedLossSequence()

    video_names_train = read_txt_file(config.videos_train_file)

    #数据集加载，RGB_and_OF是数据加载类
    #gt_dir:指定目标数据ground truth目录
    train_video360_dataset = RGB_and_OF(config.frames_dir, config.optical_flow_dir, config.gt_dir, config.frequency_dir, video_names_train, config.sequence_length, split='train', resolution=config.resolution)
    val_video360_dataset = RGB_and_OF(config.frames_dir, config.optical_flow_dir, config.gt_dir, config.frequency_dir, video_names_train, config.sequence_length, split='validation', resolution=config.resolution)

    #训练和验证集的数据被加载为批量数据
    #8个子进程数提高加载效率。shuffle随机打乱训练集的顺序
    train_data = DataLoader(train_video360_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True)
    val_data = DataLoader(val_video360_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True)


    print("Number of training sequences:", len(train_video360_dataset))
    #sample = train_video360_dataset[0]
    #print("Sample input shape:", sample[0].shape)  # 应该是 (seq_len, 43, H, W)


    print(model)
    model = train(train_data, val_data, model, device, criterion, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)

    print("Training finished")
