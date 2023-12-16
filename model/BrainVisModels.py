import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
from .layers import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock
from torch.autograd import Variable
import torch.optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.utils.backcompat.broadcast_warning.enabled = True

class TimeFreqEncoder(nn.Module):
    def __init__(self, pretrained_model_time,pretrained_model_freq,args):
        super(TimeFreqEncoder, self).__init__()

        self.pretrained_model_time = pretrained_model_time
        self.pretrained_model_time.nocliptune=True
        self.pretrained_model_time.linear_proba=False
        self.pretrained_model_freq=pretrained_model_freq

        self.fc01 =nn.Linear( args.d_model+128, args.num_class)

    def forward(self,x):
        lastrep,time_feature,cls=self.pretrained_model_time(x)
        lstmcls,freq_feature=self.pretrained_model_freq(x)
        x = torch.cat((time_feature, freq_feature), dim=1)

        lastrep = x
        encoded=x
        x = self.fc01(encoded)

        scores=x
        return lastrep,encoded,scores

class AlignNet(nn.Module):
    def __init__(self, input_size, freq_size, output_size,pretrained_model):
        super(AlignNet, self).__init__()

        self.pretrained_model = pretrained_model#TimeFreqEncoder

        self.fc01=nn.Linear(input_size+freq_size+40, 4*input_size)
        self.tanh = nn.Tanh()
        self.fc02 = nn.Linear(4*input_size, input_size)
        self.tanh = nn.Tanh()
        self.fc03=nn.Linear(input_size, 4*input_size)
        self.tanh = nn.Tanh()
        self.fc04 = nn.Linear(4*input_size, input_size)
        self.tanh = nn.Tanh()
        self.fc05=nn.Linear(input_size, 4*input_size)
        self.tanh = nn.Tanh()
        self.fc6 = nn.Linear(4*input_size, output_size)

    def forward(self, x):
        lastrep,encoded,scores=self.pretrained_model(x)
        x = torch.cat((encoded, scores), dim=1)
        x = self.fc01(x)
        x = self.tanh(x)
        res_4is_1=x
        x = self.fc02(x)
        x = self.tanh(x)
        res_is_2 = x
        x = self.fc03(x)+res_4is_1
        x = self.tanh(x)
        res_4is_2 = x
        x = self.fc04(x)+res_is_2
        x = self.tanh(x)
        x = self.fc05(x)+res_4is_2
        x = self.tanh(x)
        x = self.fc6(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter

        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x

class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        probs = self.center(x.view(-1, dim))
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]

        return indexes.view(bs, length)

class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)

        return rep_mask_token

class ChannelMapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ChannelMapping, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, int(input_dim+output_dim)//2, 1)
        self.conv2 = nn.Conv1d(int(input_dim+output_dim)//2, output_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class TimeEncoder(nn.Module):
    def __init__(self, args):
        super(TimeEncoder, self).__init__()
        d_model = args.d_model
        self.d=d_model
        self.momentum = args.momentum
        self.linear_proba = True
        self.nocliptune=True
        self.device = args.device
        self.data_shape = args.data_shape
        self.max_len = int(self.data_shape[0] / args.wave_length)
        print(self.max_len)
        self.mask_len = int(args.mask_ratio * self.max_len)
        self.position = PositionalEmbedding(self.max_len, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model, ))
        self.input_projection = nn.Conv1d(args.data_shape[1], d_model, kernel_size=args.wave_length,
                                          stride=args.wave_length)
        self.encoder = TransformerEncoder(args)
        self.momentum_encoder = TransformerEncoder(args)
        self.tokenizer = Tokenizer(d_model, args.vocab_size)
        self.reg = Regressor(d_model, args.attn_heads, 4 * d_model, 1, args.reg_layers)
        self.predict_head = nn.Linear(d_model, args.num_class)
        self.channelmapping=ChannelMapping(self.max_len,77)
        self.dimmapping = nn.Linear(d_model, 768)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def pretrain_forward(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)

        x += self.position(x)

        rep_mask_token = self.mask_token.repeat(x.shape[0], x.shape[1], 1) + self.position(x)

        index = np.arange(x.shape[1])
        random.shuffle(index)
        v_index = index[:-self.mask_len]
        m_index = index[-self.mask_len:]
        visible = x[:, v_index, :]
        mask = x[:, m_index, :]
        tokens = tokens[:, m_index]

        rep_mask_token = rep_mask_token[:, m_index, :]

        rep_visible = self.encoder(visible)
        with torch.no_grad():
            rep_mask = self.momentum_encoder(mask)

        rep_mask_prediction = self.reg(rep_visible, rep_mask_token)
        token_prediction_prob = self.tokenizer.center(rep_mask_prediction)

        return [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens]

    def forward(self, x):
        if self.linear_proba==True and self.nocliptune==True:
            #with torch.no_grad():
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            x += self.position(x)
            x = self.encoder(x)
            return torch.mean(x, dim=1)

        if self.linear_proba==False and self.nocliptune==True:
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            x += self.position(x)
            x = self.encoder(x)
            #lastrep=torch.mean(x, dim=1)
            lastrep=x
            xcls=self.predict_head(torch.mean(x, dim=1))
            return lastrep, torch.mean(x, dim=1), xcls

        if self.nocliptune == False: #CLIP
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            x += self.position(x)
            x = self.encoder(x)
            lastrep=torch.mean(x, dim=1)
            x=self.channelmapping(x)
            x = self.dimmapping(x)

            return lastrep#,x

    def get_tokens(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)
        return tokens

class FreqEncoder(nn.Module):

    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size, 40)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x = x.cpu()
        fourier_transform = np.fft.fft(x, axis=2)
        half_spectrum = fourier_transform[:, :, 1:440 // 2 + 1]
        amplitude_spectrum = np.abs(half_spectrum)

        amplitude_spectrum = torch.tensor(amplitude_spectrum).float()

        x = amplitude_spectrum.permute(0, 2, 1)
        x = x.to("cuda")

        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size),
                     torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        x = self.lstm(x, lstm_init)[0][:, -1, :]
        reps = x
        # Forward output
        xa = F.relu(self.output(x))
        x = self.classifier(xa)
        return x, xa

