#coding=utf-8
import argparse
import os
import time
import torch
import numpy  as np

from torch.nn import DataParallel
from torch.utils.data import DataLoader

from utils.logger import get_logger
import utils.exp_utils as exp_utils

from utils.exp_utils import tensor_to_cuda,write_csv
from models.survival.model import c_index
from pycox.evaluation import EvalSurv
from pycox.models.utils import pad_col
import pandas as pd
import random

import warnings
warnings.filterwarnings("ignore")
# 仅用于验证
def val(logger, cf, model, dataset):

    logger.info("performing training with model {}".format(cf.model))

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    net = model.Net(cf, logger)
    net.to(device)

    select_models = cf.select_models
    if os.path.exists(select_models):
        resume_epoch = exp_utils.load_checkpoint(select_models, net, optimizer=None)
        logger.info('resumed to checkpoint {} at epoch {}'.format(select_models, resume_epoch))

    
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)
    
    #add this , can improve the train speed
    torch.backends.cudnn.benchmark = True
    logger.info('loading dataset and initializing batch generators...')
    
    # for infer_state in ['val','test_sysucc','test_foshan','test_zhongshan','test_zhuhai']:
    for infer_state in ['val']:
        print('Inference of ', infer_state)
        with torch.no_grad():
            val_dataset = dataset.DataCustom(infer_state, cf)
            dataloaders = {}
            dataloaders['val'] = DataLoader(val_dataset,
                                            batch_size=cf.batch_size,
                                            shuffle=False,
                                            num_workers=cf.n_workers,
                                            pin_memory=True)
            logger.info("starting inference.")

            net.eval()
            
            preds = []
            mtime = []
            mevent = []
            patient_ids = []
            label_times = []
            label_states = []
            for batchidx, batch_inputs in enumerate(dataloaders['val']):
                
                ab_ct, ab_rd, ab_rs, data, mt,me,patient_id,label_time,label_state= batch_inputs
                patient_ids.extend(patient_id)
                ab_ct = tensor_to_cuda(ab_ct, device)
                ab_rd = tensor_to_cuda(ab_rd, device)
                ab_rs = tensor_to_cuda(ab_rs, device)
                data = tensor_to_cuda(data, device)
                mt = tensor_to_cuda(mt, device) #0-time, 1-state
                me = tensor_to_cuda(me, device) #0-time, 1-state
                inputs = {"ct":ab_ct, "rd":ab_rd, "rs":ab_rs, 'data':data,  'mt':mt,'me':me}
                pred = net(inputs, 'val')
                
                preds.append(pred)
                
                mtime.append(mt)
                mevent.append(me)
                label_times.append(label_time)
                label_states.append(label_state)
                
                if batchidx % 10 == 0:
                    logger.info('{}/{}'.format(batchidx, len(dataloaders['val'])))
            
            w = torch.from_numpy(np.linspace(1.0,0.5,60)).float()
            preds = torch.cat(preds, dim=0)
            mtime = torch.cat(mtime, dim=0)
            mevent = torch.cat(mevent, dim=0)
            label_times = torch.cat(label_times, dim=0)
            label_states = torch.cat(label_states, dim=0)
            mtime, mevent,label_times,label_states = mtime.squeeze(), mevent.squeeze(),label_times.squeeze(),label_states.squeeze()

            pmf = pad_col(preds.detach().cpu()).softmax(1)[:, :-1]

            risk = (pmf * w).sum(dim=1)
           
            pred_c = c_index(-risk, label_times, label_states) 

            
            cpu_preds = risk.view(1,-1)[0].detach().cpu().tolist()
            cpu_pmf = pmf.detach().cpu().tolist()
            cpu_label_times = label_times.detach().cpu().tolist()
            cpu_label_states = label_states.detach().cpu().tolist()

            total_data = np.stack([patient_ids,cpu_label_times,cpu_label_states,cpu_preds],axis=1).tolist()
            if not os.path.exists(cf.save_csv):
                os.makedirs(cf.save_csv)
            save_csv_path = os.path.join(cf.save_csv,infer_state+'.csv')
            # # # if infer_state != 'val':
            write_csv(total_data,save_csv_path,cpu_pmf, headers = ['id','time','state','risk'])             

        
        logger.info("{} C-index: {:.3f}".format(infer_state,pred_c))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='infer',
                        help='one out of : train / infer / train_infer')
    parser.add_argument('--gpu', type=str, default='6',
                        help='assign which gpu to use.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='train batch size.')
    parser.add_argument('--exp_source', type=str, default='experiments/exp_rnpc_predict',
                        help='specifies, from which source experiment to load configs, data_loader and model.')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cf = exp_utils.prep_exp(args.exp_source)
    
    if args.batch_size > 0:
        cf.batch_size = args.batch_size
    else:
        cf.batch_size = 1
    
    cf.n_workers = 0

    model = exp_utils.import_module('model', cf.model)
    dataset = exp_utils.import_module('dataset', cf.dataset_path)

    logger = get_logger(cf.log_dir)
    val(logger, cf, model, dataset)