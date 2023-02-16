import math
from pickle import NONE
from numpy.lib.function_base import append
# from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.serialization import validate_cuda_device
# from models.survival.rankloss import Cox_Loss, Cindex_loss, Regularization, Cindex_loss_queue
# from models.survival.cindex import c_index
from pycox.models.data import pair_rank_mat
from pycox.evaluation import EvalSurv
from pycox.models.utils import pad_col
import pycox

from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
import copy
from models.survival.pmf_resnet3d import generate_model
from torch.nn.parameter import Parameter


USE_TNM_MESS = True


class MLP(nn.Module):

    def __init__(self,indims_img=128, indims_cli=24, dims=128, n_classes=400):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(indims_img+indims_cli, dims),
            nn.BatchNorm1d(dims),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(dims, n_classes) 
        # self.fc = nn.Sequential(
        #     nn.Linear(dims, dims),
        #     nn.BatchNorm1d(dims),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dims, n_classes),
        # )   

    def forward(self, feat_img, cli_data):
        x = torch.cat([feat_img, cli_data], dim=1)
        x = self.stem(x)
        x = self.dropout(x)
        risk = self.fc(x)
        return risk


class Net(nn.Module):

    def __init__(self, cf=None, logger=None, K=256, m=0.98):
        super(Net, self).__init__()
        self.cf = cf
        self.logger = logger
        self.m = m
        self.num_classes = 60
        self.filters = [32, 64, 128, 128]
        self.encoder_q = generate_model(model_depth=18, planes=self.filters, n_classes=self.num_classes, n_input_channels=3)
        self.mlp = MLP(indims_img=128, indims_cli=23, dims=128, n_classes=self.num_classes)
        self.loss_func = Loss()

    def forward(self, inputs, phase='train'):
        ct, rd, rs, data, mt, me= inputs['ct'],inputs['rd'],inputs['rs'],inputs['data'],inputs['mt'],inputs['me']
        x = torch.cat([ct, rd, rs], dim=1)
        x,_ = self.encoder_q(x)
        pred = self.mlp(x, data)
        if phase=='train':
            result = self.loss_func(pred, mt, me)
            return result
        else:
            return pred  


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_func = pycox.models.loss.DeepHitSingleLoss(alpha=0.5, sigma=0.1)
        self.w = torch.from_numpy(np.linspace(1.0,0.5,60)).float()

    def forward(self, pred, mt, me):
        mt, me = mt.squeeze(), me.squeeze()
        mt[mt<0] = 0

        rank_mat = torch.tensor(pair_rank_mat(mt.cpu().numpy(), me.cpu().numpy())).to(pred.device)
        pmf_loss = self.loss_func(pred, mt.long(), me.long(), rank_mat)

        pmf = pad_col(pred.detach().cpu()).softmax(1)[:, :-1]
        surv = 1 - pmf.cumsum(1).numpy()
        eval_pred = pd.DataFrame(surv.transpose())
        ev = EvalSurv(eval_pred, mt.long().cpu().numpy(), me.cpu().numpy(), censor_surv='km')
        pmf_cindex = ev.concordance_td('antolini')

        risk = (pmf * self.w).sum(dim=1)
        sum_cindex = c_index(risk, mt, me) 
        
        result = {}
        result['loss'] = pmf_loss
        result['c-index'] = np.array(pmf_cindex)
        result['c-index2'] = sum_cindex
        result['pmf'] = torch.cat([pmf, risk[:,None]], dim=1)
        return result


def c_index(risk_pred, y, e):
    ''' Performs calculating c-index
    shape:batch size

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    risk_pred = F.sigmoid(risk_pred)
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)


if __name__ == "__main__":
    # criterion = losses.CELoss()
    # criterion = losses.CEPlusLoss()
    # criterion = losses.CoxLoss()
    # criterion = losses.CoxPlusLoss()

    ct = torch.randn((2,1, 32,64, 64)).cuda()
    rd = torch.randn((2,1, 32,64, 64)).cuda()
    rs = torch.randn((2,1,32,64, 64)).cuda()

    me = torch.tensor([1,0]).cuda()
    cli = torch.tensor([[2,0],[3,4]]).cuda()
    mt = torch.tensor([37,58]).cuda()

    model = Net().cuda()
    data = ct, rd, rs, cli, mt, me

    print(model.eval())
    results_dict = model(data)
    # loss = criterion(results_dict, survival_time,censoring,interval)
    # loss.backward()
    print(results_dict)