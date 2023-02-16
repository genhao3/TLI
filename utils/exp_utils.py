import os
import sys
import csv
import logging
import subprocess

import torch, math
import numpy as np
import pandas as pd
import importlib
from collections import OrderedDict


def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prep_exp(exp_source):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :return:
    """

    
    cf_file = import_module('cf', os.path.join(exp_source, 'configs.py'))
    cf = cf_file.cf
    cf.dataset_path = os.path.join(exp_source, 'dataset.py')

    if not os.path.exists(cf.log_dir):
        os.makedirs(cf.log_dir)

    return cf


def load_checkpoint(checkpoint_path, net, optimizer=None, is_fine_tuning=False):
    model_dict = net.state_dict()
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    pretrained_dict = checkpoint['state_dict']
    
    if is_fine_tuning:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'last_linear' not in k}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch'], checkpoint['val_c_index']


def prepare_monitoring():
    """
    creates dictionaries, where train/val metrics are stored.
    """
    # metrics = {}
    # # first entry for loss dict accounts for epoch starting at 1.
    # metrics['train'] = {'loss':[], 'class_loss': [], 'regress_loss': []}
    # metrics['val'] = {'loss':[], 'class_loss': [], 'regress_loss': []}

    metrics = {'train': {}, 'val': {}, 'infer': {}}

    return metrics


def write_csv(csv_name, content, mul=True, mod="w"):
    with open(csv_name, mod) as myfile:
        mywriter = csv.writer(myfile)
        if mul:
            mywriter.writerows(content)
        else:
            mywriter.writerow(content)


def tensor_to_cuda(tensor, device):
    if isinstance(tensor, dict):
        for key in tensor:
            tensor[key] = tensor_to_cuda(tensor[key], device)
        return tensor
    elif isinstance(tensor, (list, tuple)):
        tensor = [tensor_to_cuda(t, device) for t in tensor]
        return tensor
    else:
        return tensor.to(device,non_blocking=True)


def write_csv(total_data,save_path,cpu_pmf, headers = ['id','time','state','risk']):
    
    for i,d in enumerate(total_data):
        total_data[i].extend(list(cpu_pmf[i]))

    headers.extend(['m_'+str(x+1) for x in range(len(cpu_pmf[0]))])

    df_feature = pd.DataFrame(total_data, columns=headers)

    df_feature.to_csv(save_path,index=False)