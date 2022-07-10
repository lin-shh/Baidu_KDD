# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A GRU-based baseline model to forecast future wind power
Authors: Lu,Xinjiang (luxinjiang@baidu.com), Li,Yan (liyan77@baidu.com)
Date:    2022/03/10
"""
import torch
import torch.nn as nn


class BaselineGruModel(nn.Module):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           )
        self.projection = nn.Linear(self.hidR, self.out)

    def forward(self, x_enc):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        x = torch.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])
        x = x.cuda()
        x_enc = torch.cat((x_enc, x), 1)
        x_enc = torch.transpose(x_enc, 0, 1)
        dec, _ = self.lstm(x_enc)
        dec = torch.transpose(dec, 0, 1)
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out:]
        return sample  # [B, L, D]