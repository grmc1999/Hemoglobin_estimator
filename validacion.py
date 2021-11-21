import os
import cv2
import numpy as np
import time
import pandas as pd
df = pd.read_csv("db_sample_201901221525.csv")
tac = time.time()
def returnfactor(img, factor):
    a = int(img.shape[1]*factor)
    b = int(img.shape[0]*factor)
    return cv2.resize(img, (a, b))
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = " "
def preproces(img):
    
    a = 0
    imag_orig = cv2.imread(img)
    imgsc = returnfactor(imag_orig, 0.2)
    gray = cv2.cvtColor(imgsc, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    if eyes is not None:
        lo = img.split("\\")[-1]
        val = df.loc[df["imagename"]==lo, "ane_glo"].iloc[0]
        if len(eyes)>0 and (0<val<24):
            r_r = 0.013#0.01425
            r_eyes = []
            for (x,y,w,h) in eyes :
                if w*h>r_r*imgsc.shape[0]*imgsc.shape[1]:
                    r_eyes.append([x, y, w, h])
            if len(r_eyes)>0:
                lower = [160, 130, 0]
                upper = [210, 200, 30]
                
                # create NumPy arrays from the boundaries
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")
                # find the colors within the specified boundaries and apply
                # the mask
                mask = cv2.inRange(imgsc, lower, upper)
                output = cv2.bitwise_and(imgsc, imgsc, mask = mask)
                aa = 0
                l = []
                for numi, i in enumerate(mask):
                    for numj,j in enumerate(i):
                        if j == 255:
                            l.append([numi, numj])
                            aa -=-1
                if len(l)>0:
                    l = np.array(l, dtype="float").reshape(-1, 2)
                    
                    ee = np.mean(l, axis=0)
                    row = int(ee[0])
                    col = int(ee[1])
                    c_c = [col, row]
                    for (x, y, w, h) in r_eyes:
                        c_i = [int(x+w/2), int(y+h/2)]
                        if abs(c_i[1]-c_c[1])<=int(1.2*h) and abs(c_i[0]-c_c[0])<=int(1.2*w/2):
                            print(f"{lo} valida")
                            return True
                print(f"{lo} invalida")
                return False  
            print(f"{lo} invalida")
            return False  
        print(f"{lo} invalida")
        return False
    
    print(f"{lo} invalida")
    return False

directory = "C:\\UNI\\Labo\\Liiarpi_1-master\\Imagenes_Originales"
i = 0
j = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        val = open("valido.txt", "a")
        ival = open("novalido.txt", "a")

        img =os.path.join(directory, filename)
        ff = preproces(img)
        if ff:
            val.write(filename)
            val.write("\n")
            i +=1
        else:
            ival.write(filename)
            ival.write("\n")
            j-=-1
val.close()
ival.close()
print(f"{i} imagenes validas \n {j} imagenes invalidas")
        
print(time.time()- tac)