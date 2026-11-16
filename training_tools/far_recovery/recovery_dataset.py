import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class RecoveryDataset(Dataset):
    def __init__(self,root):
        self.root=root
        with np.load(root/'data.npz') as data:
            self.data={key:data[key].copy() for key in data.files}
    def __len__(self):return len(self.data['state'])
    def __getitem__(self,index):
        row={key:torch.from_numpy(self.data[key][index].copy()) for key in ('state','action','action_is_pad')}
        row['observation.state']=row.pop('state')
        for key,path in zip(('head','left_wrist','right_wrist'),self.data['images'][index]):
            image=cv2.cvtColor(cv2.imread(str(self.root/path)),cv2.COLOR_BGR2RGB)
            row['observation.images.'+key]=torch.from_numpy(image.transpose(2,0,1).copy()).float()/255
        return row
