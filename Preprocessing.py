import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
def imadjust(img, In=(0.0,255.0), Out=(0,255.0), gamma=1.0):
    "J = low_out +(high_out - low_out).* ((I - low_in)/(high_in - low_in)).^ gamma"
    low_in,high_in = In
    low_out, high_out = Out
 
   
    k = (high_out - low_out) / (high_in - low_in)
         # Gamma transformation table
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    h,w = img.shape[:2]
    imgOut = np.zeros((h,w), np.uint8)
    
    for r in range(h):
        for c in range(w):
            if img[r,c] <= low_in:
                imgOut[r,c] = low_out                
            elif img[r,c] > high_in:
                imgOut[r,c] = high_out
            else:
                res = int(k*(img[r,c]-low_in) + low_out)
                imgOut[r,c] = table[res]#Check table
               
    return imgOut
m = sys.argv[1]
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread(m)

print(img.shape)
try:
    factor = float(sys.argv[2])
except:
    factor = 0.2
def returnfactor(img, factor):
    a = int(img.shape[1]*factor)
    b = int(img.shape[0]*factor)
    return cv2.resize(img, (a, b))
imgsc = returnfactor(img, factor)
gray = cv2.cvtColor(imgsc, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray)
a = []
ma = -100
fl = -1
b = []
print(eyes)
if eyes.shape[0]>1 :
    for i, (x,y,w,h) in enumerate(eyes) :
        if w*h > ma:
            ma = max(ma, w*h)
            fl = i
            b = [x,y,w,h]
else:
    fl = 0
    b = eyes[0]
print(fl, b)


for (x,y,w,h) in eyes :

    print(x,y,w,h)
    a.append([int(x/factor),int(y/factor),int(w/factor),int(h/factor)])
    cv2.rectangle(imgsc,(x,y),(x+w,y+h),(255,0,0),2)
print(len(eyes))
cv2.imshow('img', imgsc)
#print(a)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()


fimg = img[int(b[1]/factor):int(b[1]/factor)+ int(b[3]/factor),int(b[0]/factor):int(b[0]/factor)+ int(b[2]/factor),:]
rrc = np.array(fimg)
brown_lo=np.array([200,200,140])
brown_hi=np.array([255,255,255])

cv2.imshow('fimg', returnfactor(fimg, 0.2))

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

##==================Suavizador
blur = cv2.GaussianBlur(fimg,(33,33),0)
#blur = cv2.medianBlur(blur,15)

cv2.imshow('blur', returnfactor(blur, 0.2))
print(np.max(blur))
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
#print(blur)
simg = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)[:,:,1]
arrb = np.asarray(simg)
simg=imadjust(simg,(arrb.min(),arrb.max()),(0,255))
print("MAX y min", arrb.min(), arrb.max(), sep=" ")
aux = returnfactor(simg, 0.2)
print("Intensidad ",aux.shape)
cv2.imshow('image_out', aux)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
'''otsu = np.zeros(blur.shape)
otsu[blur>=128]=0
otsu[blur<128]=1
cv2.imshow('otsu', returnfactor(otsu, 0.2))
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
'''

#===========Threshold===============
raux = np.asarray(aux).reshape(-1,1)
print("Media", np.mean(raux))
n_tr = int(np.mean(raux)+1.5*np.std(raux))
ret, thresh = cv2.threshold(simg, n_tr, 255, 0)
cv2.imshow('otsu', returnfactor(thresh, 0.2))
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("========Contours=========")
print(len(contours))
def hola(ccc):
    aa = thresh.shape[0]/2
    M = cv2.moments(ccc)
    cy = int(M['m01']/(M['m00']))
    cx = int(M['m10']/(M['m00']))
    return np.sqrt((cx-aa)**2+(cy-aa)**2)
con2 = sorted(contours, key=cv2.contourArea)
rcon3 = con2[-2:]
con3 = sorted(rcon3, key = hola)
ccon = con3[0]

fimg = cv2.drawContours(fimg, contours, -1, (0, 255, 0), 8)
cv2.imshow('edges', returnfactor(fimg, 0.2))
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

mask = np.zeros_like(fimg)
out = np.zeros_like(fimg)
print(mask.shape, out.shape, sep='\n')
mask = cv2.fillPoly(mask, pts =[ccon], color=(255,255,255))
out[mask == 255] = rrc[mask == 255]
cv2.imshow('output', returnfactor(out, 0.2))
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

