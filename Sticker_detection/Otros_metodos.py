import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage.color import rgb2hsv
from skimage.morphology import convex_hull_image
from sklearn.decomposition import PCA
from skimage import measure

#Método de Selección por limite adaptativo

def cadapth(img,ith=0.8,maxiter=25,coe=0.035,step=0.01,w=0.04,initial_cut=3.5):
  hsv_i= rgb2hsv(img)
  med = cv2.blur(hsv_i,(25,25),0)
  val_img = med[:, :, 2]
  cimg=ucut(val_img,initial_cut)
  binary_img = np.logical_and((cimg > ith),((ith+w) > cimg))
  m=Assem(val_img,binary_img,initial_cut)
  con=convex_hull_image(m)
#  print(ith)
#  print(cmask(con)/(len(con.ravel())))
  if (cmask(con)/(len(con.ravel())))>coe:
    for i in range(maxiter):
      ith=ith+step
#      print(ith)
      binary_img=np.logical_and((cimg > ith),((ith+w) > cimg))
      m=Assem(val_img,binary_img,initial_cut)
      con=convex_hull_image(binary_img)
#      print(cmask(con)/(len(val_img.ravel())))
      if ((cmask(con)/(len(val_img.ravel())))<coe)or(ith>=1):
        break
  return ith

  def adapth(img,ith=0.8,f=3,maxiter=25,coe=0.035,step=0.01,w=0.04):

  hsv_i= rgb2hsv(img)
  med = cv2.blur(hsv_i,(25,25),0)
  val_img = med[:, :, 2]
  binary_img = np.logical_and((val_img > ith),((ith+w) > val_img))
  con=convex_hull_image(binary_img)

#  print(ith)
#  print(cmask(con)/(len(con.ravel())))
  if (cmask(con)/(len(con.ravel())))>coe:
    for i in range(maxiter):
      ith=ith+step
#      print(ith)
      binary_img=np.logical_and((val_img > ith),((ith+w) > val_img))
      con=convex_hull_image(binary_img)
#      print(cmask(con)/(len(con.ravel())))
      if ((cmask(con)/(len(con.ravel())))<coe)or(ith>=1):
        break
  return ith