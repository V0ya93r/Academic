import Modules
import torch.nn as nn
import torch


class SST_Sal(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=36, output_dim=1):
        super(SST_Sal, self).__init__()

        self.encoder = Modules.SpherConvLSTM_EncoderCell(input_dim, hidden_dim)
        self.decoder = Modules.SpherConvLSTM_DecoderCell(hidden_dim, output_dim)


    def forward(self, x):

        b, _, _, h, w = x.size()
        state_e = self.encoder.init_hidden(b, (h, w))
        state_d = self.decoder.init_hidden(b, (h//2, w//2))


        outputs = []

        for t in range(x.shape[1]):
            out, state_e = self.encoder(x[:, t, :, :, :], state_e)
            out, state_d = self.decoder(out, state_d)
            outputs.append(out)
        return torch.stack(outputs, dim=1)


"""
import Modules
import torch.nn as nn
import torch

class SST_Sal(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[36, 72], output_dim=1):
        
        #多层 Encoder-Decoder 结构
        #input_dim: 输入通道 (RGB+Flow+Freq = 9)
        #hidden_dims: 每一层的隐藏维度 [encoder1, encoder2]
        #output_dim: 输出通道 (显著性图 = 1)
        
        super(SST_Sal, self).__init__()

        # Encoder 两层
        self.encoder1 = Modules.SpherConvLSTM_EncoderCell(input_dim, hidden_dims[0])
        self.encoder2 = Modules.SpherConvLSTM_EncoderCell(hidden_dims[0], hidden_dims[1])

        # Decoder 两层
        self.decoder2 = Modules.SpherConvLSTM_DecoderCell(hidden_dims[1], hidden_dims[0])
        self.decoder1 = Modules.SpherConvLSTM_DecoderCell(hidden_dims[0], output_dim)

    def forward(self, x):
        b, _, _, h, w = x.size()

        # 初始化 hidden state
        state_e1 = self.encoder1.init_hidden(b, (h, w))
        state_e2 = self.encoder2.init_hidden(b, (h // 2, w // 2))
        state_d2 = self.decoder2.init_hidden(b, (h // 4, w // 4))
        state_d1 = self.decoder1.init_hidden(b, (h // 2, w // 2))

        outputs = []

        for t in range(x.shape[1]):
            # Encoder 第一层
            out1, state_e1 = self.encoder1(x[:, t, :, :, :], state_e1)
            # Encoder 第二层
            out2, state_e2 = self.encoder2(out1, state_e2)

            # Decoder 第二层
            d2, state_d2 = self.decoder2(out2, state_d2)
            # Skip connection: 融合 encoder1 的输出
            d2 = d2 + out1

            # Decoder 第一层
            d1, state_d1 = self.decoder1(d2, state_d1)

            outputs.append(d1)

        return torch.stack(outputs, dim=1)
"""