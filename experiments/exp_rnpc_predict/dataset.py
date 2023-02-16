import torch.utils.data as data
import pandas as pd
import numpy as np
import os
from models.survival.discretization import DiscretizeUnknownC

CT_MIN_VALUE = -100
CT_MAX_VALUE = 200

RD_MIN = 0
RD_MAX = 80



class DataCustom(data.Dataset):
    def __init__(self,state,cf):
        self.state = state
        self.input_size = cf.input_size
        self.train_npz_path = cf.train_npz_path
        self.validation_npz_path = cf.validation_npz_path 

        if self.state == 'train':
            csv_path = cf.train_csv
            self.data_dir = self.train_npz_path 
        elif self.state == 'val':
            csv_path = cf.val_csv
            self.data_dir = self.validation_npz_path 
        

        self.df = pd.read_csv(csv_path)
        self.ids= self.df[self.df['M']==0].loc[:,'id']
        self.df = self.df.set_index('id')
        self.cli_data = self.df.loc[:,['sex','age','T','N','cure_method']] #'T','N','cure_method'
        self.cut_off_age_6_ground()
        self.dvh_data = self.df.loc[:,'D0_5cc':'TLV']
        self.normalize_dvh()
        self.label = self.df.loc[:,['time','state']]
        
        #check
        self.path_dict = {}
        self.uids = []
        for id in self.ids:
            if not isinstance(self.data_dir, (list, tuple)):
                self.data_dir = [self.data_dir]
            for subdir in self.data_dir:
                path = os.path.join(os.path.join(subdir, id+'.npz'))
                if os.path.exists(path):
                    self.path_dict[id] = path
                    self.uids.append(id)
                    break
        print('Num of uids for {}: {}'.format(self.state, len(self.uids)))

        self.labtrans = DiscretizeUnknownC(np.array(cf.points).astype(np.float32), right_censor=True)


    def cut_off_age_6_ground(self):
        '''
        ≤20,21-30,31-40,41-50,51-60，＞60
        '''
        self.cli_data['age_6_ground'] = ''
        self.cli_data.loc[self.cli_data.loc[:,'age']<=20,'age_6_ground']=0
        self.cli_data.loc[(self.cli_data.loc[:,'age']>20) & (self.cli_data.loc[:,'age']<=30),'age_6_ground']=1
        self.cli_data.loc[(self.cli_data.loc[:,'age']>30) & (self.cli_data.loc[:,'age']<=40),'age_6_ground']=2
        self.cli_data.loc[(self.cli_data.loc[:,'age']>40) & (self.cli_data.loc[:,'age']<=50),'age_6_ground']=3
        self.cli_data.loc[(self.cli_data.loc[:,'age']>50) & (self.cli_data.loc[:,'age']<=60),'age_6_ground']=4
        self.cli_data.loc[self.cli_data.loc[:,'age']>60,'age_6_ground']=5


    def load_npz(self,npz_path):
        dict_data = np.load(npz_path,allow_pickle=True)
        ct = dict_data['ct'].astype(np.float32).squeeze()
        rs = dict_data['rs'].astype(np.float32).squeeze()
        rd = dict_data['rd'].astype(np.float32).squeeze()

        return ct, rd, rs


    def random_crop(self,ct, rd, rs):
        '''
        function:train is random crop,val is center crop
        '''
        d, h, w = ct.shape[0], ct.shape[1], ct.shape[2]
        i_d, i_h, i_w = self.input_size[0], self.input_size[1], self.input_size[2]

        if self.state == 'train':
            # random crop
            e_d = d - i_d
            e_h = h - i_h
            e_w = w - i_w

            s_d = np.random.randint(0, e_d+1)
            s_h = np.random.randint(0, e_h+1)
            s_w = np.random.randint(0, e_w+1)

            ct = ct[s_d:s_d+i_d, s_h:s_h+i_h, s_w:s_w+i_w]
            rd = rd[s_d:s_d+i_d, s_h:s_h+i_h, s_w:s_w+i_w]
            rs = rs[s_d:s_d+i_d, s_h:s_h+i_h, s_w:s_w+i_w]
            

        elif self.state != 'train':
            # center crop
            center_d, center_h, center_w =d // 2, h // 2, w // 2
            step_d, step_h, step_w = i_d // 2, i_h // 2, i_w // 2
            ct = ct[center_d - step_d:center_d + step_d, center_h - step_h:center_h + step_h,center_w - step_w:center_w + step_w]
            rd = rd[center_d - step_d:center_d + step_d, center_h - step_h:center_h + step_h,center_w - step_w:center_w + step_w]
            rs = rs[center_d - step_d:center_d + step_d, center_h - step_h:center_h + step_h,center_w - step_w:center_w + step_w]
        
        return ct, rd, rs


    def normalize_dvh(self):
        vmin = np.array([ 0,  0.,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype(np.float32)
        vmax = np.array([ 100.,  100.,1,1,1,1,1,1,1,1,1,1,1,1,100,20,60,200]).astype(np.float32)
        self.dvh_data = (self.dvh_data-vmin)/(vmax-vmin)
        


    def normalized(self,img,low,hight):
        # 0-1
        img = np.clip(img, low, hight)
        img = (img - low) / (hight - low)
        return img

    def random_flip_func(self, sample, rd, mask, dims=[0, 1, 2]):
        '''
        sample: 3d-array
        '''
        random_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
        flipid = [1, 1, 1]
        for dim in dims:
            flipid[dim] = random_id[dim]

        # ascontiguousarray
        sample = np.ascontiguousarray(sample[::flipid[0],::flipid[1],::flipid[2]])
        rd = np.ascontiguousarray(rd[::flipid[0],::flipid[1],::flipid[2]])
        mask = np.ascontiguousarray(mask[::flipid[0],::flipid[1],::flipid[2]])

        return sample, rd, mask

    def __getitem__(self, idx):
        fn = self.uids[idx]
        
        path = self.path_dict[fn]
        ct, rd, rs = self.load_npz(path)
        # crop
        ct, rd, rs = self.random_crop(ct, rd, rs)
        # flip
        if self.state == 'train':
            ct, rd, rs = self.random_flip_func(ct, rd, rs, dims=[0,1,2])

        ct = self.normalized(ct, CT_MIN_VALUE,CT_MAX_VALUE)
        rd = self.normalized(rd, RD_MIN, RD_MAX)

        ct = ct[np.newaxis, ...]
        rd = rd[np.newaxis, ...]
        rs = rs[np.newaxis, ...]

        cli = self.cli_data.loc[fn,['sex','age_6_ground','T','N','cure_method']].values.astype(np.float32)
        dvh = self.dvh_data.loc[fn].values.astype(np.float32)
        data = np.concatenate([cli, dvh], axis=0).astype(np.float32)

        label = self.label.loc[fn].values
        label = label.astype(np.float32)
        mt, me = self.labtrans.transform(label[:1], label[1:])
        mt = mt-1
        if mt[0] == -np.inf:
            mt[0] = 0

        return ct, rd, rs, data, mt, me, fn,label[0],label[1]

    def __len__(self):
        return len(self.uids)