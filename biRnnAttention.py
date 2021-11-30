# -*- coding: utf-8 -*-
# !@time: 2021/12/12 下午10:46
# !@author: superMC @email: 18758266469@163.com
# !@fileName: biRnnAttention.py


import torch
import torch.nn as nn
from markdown.util import deprecated
from torch.autograd import Variable
import torch.nn.functional as F





class BiLstmAttention(torch.nn.Module):
    def __init__(self, batch_size, input_size, output_size, bidirectional, dropout,
                 attention_size, sequence_length, use_cuda=True):
        super(BiLstmAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length

        self.layer_size = 1
        self.lstm = nn.LSTM(self.input_size,
                            self.output_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size
        self.hidden_size = self.output_size
        self.attention_size = attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, x):

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(x)
        attn_output = self.attention_net(lstm_output)

        return attn_output
