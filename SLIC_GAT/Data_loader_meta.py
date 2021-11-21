from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def read_csv(path,n):
      landmarks_frame = pd.read_csv(path)
      img_name = landmarks_frame.iloc[n, 0]
      landmarks = landmarks_frame.iloc[n, 1:]
      landmarks = np.asarray(landmarks)
      landmarks = landmarks.astype('float').reshape(-1, 2)
      return landmarks


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y


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