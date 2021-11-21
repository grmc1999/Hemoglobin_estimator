from __future__ import print_function, division
import os

from numpy.lib.function_base import _SIGNATURE
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

def batch_graphs_(gs):
    NUM_FEATURES = gs[0][0].shape[-1]
    G = 1
    N=gs[0].shape[0]
    M=gs[1].shape[0]
    adj = np.zeros([N,N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N,M])
    Mtgt = np.zeros([N,M])
    Mgraph = np.zeros([N,G])
    h = np.concatenate([g[0] for g in gs])
    
    n_acc = 0
    m_acc = 0
    #for g_idx, g in enumerate(gs):
    n = gs[0].shape[0]
    m = gs[1].shape[0]
        
    for e,(s,t) in enumerate(gs[1]):
        adj[int(n_acc+s),int(n_acc+t)] = 1
        adj[int(n_acc+t),int(n_acc+s)] = 1
        
        src[int(m_acc+e)] = n_acc+s
        tgt[int(m_acc+e)] = n_acc+t
            
        Msrc[int(n_acc+s),int(m_acc+e)] = 1
        Mtgt[int(n_acc+t),int(m_acc+e)] = 1
            
    Mgraph[int(n_acc):int(n_acc+n),0] = 1
        
    #n_acc += n
    #m_acc += m
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE)
    )

def read_csv(path,n):
      landmarks_frame = pd.read_csv(path)
      img_name = landmarks_frame.iloc[n, 0]
      landmarks = landmarks_frame.iloc[n, 1:]
      landmarks = np.asarray(landmarks)
      landmarks = landmarks.astype('float').reshape(-1, 2)
      return landmarks


def show_landmarks(image, landmarks):
      plt.imshow(image)
      plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
      plt.pause(0.001)  # pause a bit so that plots are updated

def sample_central_simple(h,e,n):
    h=np.hstack((h, np.linalg.norm(h[:,[-5,-3]]-np.mean(h[:,[-5,-3]],axis=0),axis=1).reshape(-1,1)))
    h=np.hstack((h,np.arange(0,h.shape[0]).reshape(-1,1)))
    
    h = np.array(sorted(list(h), key=lambda ni: ni[-2], reverse=False))[:n]
    h = np.array(sorted(list(h), key=lambda ni: ni[-1], reverse=False))
    mp=np.hstack((h[:,-1].reshape(-1,1),np.arange(0,h.shape[0]).reshape(-1,1)))
    
    t=np.vectorize(lambda ni,ei:ei in ni,signature="(j),()->()")(h[:,-1],e.reshape(-1,)).reshape(-1,2)
    e=e[np.prod(t,axis=1).astype(bool)]
    e=np.vectorize(lambda mp,ei:mp[np.equal(mp[:,0],ei),1],
                   signature='(j,i),()->()')(mp,e.reshape(-1,)).reshape(-1,2)
    h[:,-1]=np.arange(0,h.shape[0])
    h=h[:,:-2]
    return h,e



class Dataset(torch.utils.data.Dataset):
      def __init__(self, root_dir,medata_dir,transform=None):
            'Initialization'
            #directories
            self.root_dir = root_dir
            self.medata_dir=medata_dir

            #Read names
            raw_data=np.array(os.listdir(os.path.join(root_dir,"Sem_Auto1")))
            self.raw_data=raw_data
            meta_data=np.array(os.listdir(os.path.join(root_dir,medata_dir)))
            self.meta_data=meta_data

            #only graph
            grph_data=meta_data[np.vectorize(lambda dat:dat.split(".")[-1]=="npy")(meta_data)]
            name_grph_data=np.vectorize(lambda dat: ("_").join(dat.split("_")[:1]))(grph_data)
            uniq_grph_data=np.unique(name_grph_data)

            #Avoid non proccesed
            pr_names=np.vectorize(lambda dat:(".").join(dat.split(".")[:-1]))(uniq_grph_data)
            npr_names=np.vectorize(lambda dat:("_").join(dat.split("_")[1:-1]))(raw_data)
            raw_data=raw_data[np.vectorize(lambda pr,dat:dat in pr,signature="(j),()->()")(pr_names,npr_names)]

            y=np.vectorize(lambda dat:float((".").join(dat.split(".")[:-1]).split("_")[-1]))(raw_data).reshape(-1,1)
            self.landmarks_frame=np.hstack((grph_data.reshape(-1,1),y.reshape(-1,1)))
            self.transform = transform

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.landmarks_frame)

      def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample
            if torch.is_tensor(idx):
                  idx=idx.tolist()

            img_name = os.path.join(self.root_dir,self.landmarks_frame[idx, 0])
            data_name=self.root_dir+"\\"+self.medata_dir+"\\"+(lambda d:("_").join(d.split('.')[0].split('_')[1:-1])+".npy")(self.landmarks_frame[idx, 0])
            #image = io.imread(img_name)
            data=np.load(data_name,allow_pickle=True)
            landmarks = self.landmarks_frame[idx, 1:]
            landmarks = np.array([landmarks])

            landmarks = landmarks.astype('float')
            sample = {'image_graph': data, 'landmarks': landmarks}
            if self.transform:
                  sample=self.transform(sample)

            return sample

class Rotated_Dataset(torch.utils.data.Dataset):
      def __init__(self, root_dir,medata_dir,features=np.arange(0,62),range=[0,25],transform=None):
            'Initialization'
            #directories
            self.root_dir = root_dir
            self.medata_dir=medata_dir
            self.features=features

            #Read names
            #raw_data=np.array(os.listdir(root_dir+"\\Sem_Auto1"))
            raw_data=np.array(os.listdir(os.path.join(root_dir,"Sem_Auto1")))
            self.raw_data=raw_data
            #meta_data=np.array(os.listdir(root_dir+"\\"+medata_dir))
            meta_data=np.array(os.listdir(os.path.join(root_dir,medata_dir)))
            self.meta_data=meta_data

            #only graph
            grph_data=meta_data[np.vectorize(lambda dat:dat.split(".")[-1]=="npy")(meta_data)]
            name_grph_data=np.vectorize(lambda dat: ("_").join(dat.split("_")[1:]))(grph_data)
            uniq_grph_data=np.unique(name_grph_data)

            #Avoid non proccesed
            pr_names=np.vectorize(lambda dat:(".").join(dat.split(".")[:-1]))(uniq_grph_data)
            npr_names=np.vectorize(lambda dat:("_").join(dat.split("_")[1:-1]))(raw_data)
            raw_data=raw_data[np.vectorize(lambda pr,dat:dat in pr,signature="(j),()->()")(pr_names,npr_names)]
            npr_names=np.vectorize(lambda dat:("_").join(dat.split("_")[1:-1]))(raw_data)

            iy=np.vectorize(lambda dat:float((".").join(dat.split(".")[:-1]).split("_")[-1]))(raw_data).reshape(-1,)

            #Broadcast y
            y=np.zeros(name_grph_data.shape).reshape(-1,)
            name_grph_data=np.vectorize(lambda dat:('.').join(dat.split('.')[:-1]))(name_grph_data)
            t=np.vectorize(self.build_tar,signature="(i),(),(j)->(w)")(npr_names,name_grph_data,iy)
            t=np.hstack((grph_data.reshape(1,-1).T,t))

            self.landmarks_frame=np.delete(t,1,axis=1)
            #Borrar 0's
            #self.landmarks_frame=np.delete(self.landmarks_frame,np.where(self.landmarks_frame[:,1].astype(float)==0),axis=0)
            self.landmarks_frame=np.delete(self.landmarks_frame,np.where(self.landmarks_frame[:,1].astype(float)>20),axis=0)
            self.landmarks_frame=np.delete(self.landmarks_frame,np.where(self.landmarks_frame[:,1].astype(float)<5),axis=0)

            #Select range in Y
            sel=np.logical_and(self.landmarks_frame[:,1].astype(float)>range[0],self.landmarks_frame[:,1].astype(float)<range[1])
            self.landmarks_frame=self.landmarks_frame[sel]
            
            self.transform = transform

      #def build_tar(self,tar,dat,names,graphs,y):
      def build_tar(self,names,graphs,y):
            #print(graphs)
            return np.array([graphs,str(y[graphs==names][0])])

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.landmarks_frame)

      def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample
            if torch.is_tensor(idx):
                  idx=idx.tolist()

            img_name = os.path.join(self.root_dir,self.landmarks_frame[idx, 0])
            data_name=os.path.join(self.root_dir,self.medata_dir,self.landmarks_frame[idx, 0])
            data=np.load(data_name,allow_pickle=True)
            #print(data)
            data[0]=data[0][:,self.features]


            landmarks = self.landmarks_frame[idx, 1:]
            landmarks = np.array([landmarks])

            landmarks = landmarks.astype('float')
            sample = {'image_graph': data, 'landmarks': landmarks}
            if self.transform:
                  sample=self.transform(sample)
            return sample

class Dataset_direct(torch.utils.data.Dataset):
      def __init__(self,root_dir,transform=None):
            'Initialization'
            self.landmarks_frame = np.array(os.listdir(root_dir))
            self.transform = transform

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.landmarks_frame)

      def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample
            if torch.is_tensor(idx):
                  idx=idx.tolist()

            img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx, 0])
            image = io.imread(img_name)
            landmarks = self.landmarks_frame.iloc[idx, 1:]
            landmarks = np.array([landmarks])

            landmarks = landmarks.astype('float').reshape(-1, 2)
            sample = {'image': image, 'landmarks': landmarks}
            if self.transform:
                  sample=self.transform(sample)

            return sample


class Rescale(object):
      def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size

      def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']

            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                  if h > w:
                        new_h, new_w = self.output_size * h / w, self.output_size
                  else:
                        new_h, new_w = self.output_size, self.output_size * w / h
            else:
                  new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
            landmarks = landmarks * [new_w / w, new_h / h]

            return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
      def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                  self.output_size = (output_size, output_size)
            else:
                  assert len(output_size) == 2
                  self.output_size = output_size

      def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']

            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top: top + new_h,
                      left: left + new_w]

            landmarks = landmarks - [left, top]

            return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
      def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

#class applySLIC(object):

def show_landmarks_batch(sample_batched):

      images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
      batch_size = len(images_batch)
      im_size = images_batch.size(2)      
      grid = utils.make_grid(images_batch)
      plt.imshow(grid.numpy().transpose((1, 2, 0)))

      for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                  landmarks_batch[i, :, 1].numpy(),
                  s=10, marker='.', c='r')

      plt.title('Batch from dataloader')