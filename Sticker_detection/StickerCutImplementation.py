from skimage.io import imsave
from Utilities import *
import os

di='/home/grmc1999/Documentos/LIIARPI/hmg/TempDBR/Original'

imgs=os.listdir(di)

for imgd in imgs:
    ddi=di+'/'+imgd
    img = io.imread(ddi)
    try:
        ROI1=stickerCut(img)
        n=ddi.split('/')
        name="SF_"+n[-1]
        dird=('/').join(n[0:-2])
        dird=dird+'/Imagenes_recortadas_por_sticker'
        imsave(dird+'/'+name, ROI1)
    except:
        print(ddi)