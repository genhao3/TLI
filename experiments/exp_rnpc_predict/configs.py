# coding=utf-8
import os
import numpy as np
from easydict import EasyDict as edict


cf = edict()
#model setting
cf.infer_model= 'whole' #'patches' or 'whole'
cf.model_mode = 'no_anchor'
cf.mode_dl= 'classification' # 'segmentation', 'classification', 'detection'
cf.model = 'models/survival/model.py'

cf.output_dir = './output'
cf.train_npz_path = "data_example/npz/train/"
cf.validation_npz_path = "data_example/npz/validation/"


cf.train_csv = "data_example/csv/train.csv"
cf.val_csv = "data_example/csv/val.csv"

cf.n_workers = 8
cf.input_channels = 1
cf.num_classes = 1
cf.groups = 32
cf.input_size = (32, 64, 64)  # z y x
cf.points = [i+1 for i in range(60)]




cf.log_dir = os.path.join(cf.output_dir, 'logs')
cf.select_models = 'models/weights/model_030.tar'
cf.save_csv = os.path.join(cf.output_dir, 'csv_result')