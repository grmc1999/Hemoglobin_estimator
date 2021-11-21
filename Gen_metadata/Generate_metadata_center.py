import sys
import pathlib
pth=str(pathlib.Path().absolute())
sys.path.append(('\\').join(pth.split('\\')[:-1])+"\\Utils")
from Utilities import *
import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage import morphology
import networkx as nx
import json

f1=open(('\\').join(pth.split('\\')[:-2])+"\\Data_base\\validcrop_1.txt","r")
lines=f1.readlines()
linesn=np.array(lines)
linesn=np.delete(lines,np.where(linesn=="\n"))
linesn=linesn.reshape(-1,3)
linesnc=v_replace_err(linesn)
linesnc=np.vectorize(pyfunc=lambda x:np.array([x[0].split('\n')[0]]),signature="(n)->(m)")(linesnc.reshape(-1,1)).reshape(-1,3)
xywh=linesnc[:,:2]
imgnames=linesnc[:,2]
xywh=v_no_spaces(xywh)
xywh=np.vectorize(pyfunc=(lambda x:float(x)))(xywh.reshape(1,-1)[0])
xywh=xywh.reshape(-1,4).astype(int)+1
print(imgnames)

dir_origin=('\\').join(pth.split('\\')[:-2])+'\\Data_base\\Imagenes_originales\\'
dir_ROI=('\\').join(pth.split('\\')[:-2])+'\\Data_base\\Sem_Auto\\eye_'
dir_meta=('\\').join(pth.split('\\')[:-2])+'\\Data_base\\Metadata_V5G\\'
#FOR
for name in imgnames:
    try:
        img = io.imread(dir_origin+name)
        ROI = io.imread(dir_ROI+name)
        mask=assemble_mask(xywh[np.where(imgnames==(name))][0],img,ROI)
#           lum = np.mean(mask,axis=2).astype(int)
#           mask1=lum > 0
        SD,G,h,edges=get_graph_from_image(img,mask,desired_nodes=20)
        
        #FOR Pytorch
        np.save(dir_meta+name.split('.')[0]+'.npy',np.array([h,edges]))
        nx.readwrite.adjlist.write_adjlist(G,dir_meta+name.split('.')[0])

        #FOR Sckit
        #Sampled=sample_central(SD,G,num=5,maxdeg=3)
        #NSD={}
        #for ID in Sampled:
        #    NSD[ID]=SD[ID]

        print("done")
    except Exception as e:
    #    exc_type, exc_obj, exc_tb = sys.exc_info()
    #    print(exc_type, exc_tb.tb_lineno)
        print(name)
        print(e)

def apply(dir_origin,dir_ROI,dir_meta,name):
    try:
        img = io.imread(dir_origin+name)
        ROI = io.imread(dir_ROI+name)
        mask=assemble_mask(xywh[np.where(imgnames==(name))][0],img,ROI)
#        lum = np.mean(mask,axis=2).astype(int)
#        mask1=lum > 0
        #SD,segments=get_Statistical_Descriptors_(img,mask,n_segments=20)
        SD,G,h,edges=get_graph_from_image(img,mask,desired_nodes=20)
        Sampled=sample_central(SD,G,num=5,maxdeg=3)
        NSD={}
        for ID in Sampled:
            NSD[ID]=SD[ID]



        np.save(dir_meta+name.split('.')[0]+'.npy',NSD)
        print("done")
        return 0
    except Exception as e:
        print(name)
        print(e)
        return 0
v_apply=np.vectorize(apply,signature="(),(),(),()->()")
#v_apply(dir_origin,dir_ROI,dir_meta,imgnames)
#v_apply(dir_origin,dir_ROI,dir_meta,np.array(imgnames))
#print(type(imgnames))