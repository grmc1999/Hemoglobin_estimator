import cv2
import numpy as np
import sys
m = sys.argv[1]
img = cv2.imread(m)
factor = 0.2
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def returnfactor(img, factor):
    a = int(img.shape[1]*factor)
    b = int(img.shape[0]*factor)
    return cv2.resize(img, (a, b))
image =returnfactor(img, factor)
'''boundaries = [
	([150, 213, 0], [175, 253, 22]),
	([180, 155, 0], [195, 175, 5])
]'''
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray)
if not eyes is None:
    r_r = 0.01425
    r_eyes = []
    for (x,y,w,h) in eyes :
        print(w*h, image.shape[0]*image.shape[1])
        if w*h>r_r*image.shape[0]*image.shape[1]:
            r_eyes.append([x, y, w, h])
    if len(r_eyes)>0:
        lower = [160, 130, 0]
        upper = [210, 200, 30]
        
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        aa = 0
        l = []
        for numi, i in enumerate(mask):
            for numj,j in enumerate(i):
                if j == 255:
                    l.append([numi, numj])
                    aa -=-1

        if len(l)>0:
            l = np.array(l, dtype="float").reshape(-1, 2)
            '''
            #Pruebaaaaaaa
            for jj in range(1,5):
                aaaa = []
                esa, ese = np.mean(l[:,0]), np.mean(l[:,1])
                es_a, es_e = np.std(l[:,0]), np.std(l[:,1])
                for i in l:
                    if np.abs(i[0]-esa)<=jj*es_a*1.1 and np.abs(i[0]-esa)<=jj*es_a*1.1:
                        aaaa.append(i)
                l = np.array(aaaa, dtype="float").reshape(-1, 2)
            #Pruebaaaaaaaaa
            '''
            ee = np.mean(l, axis=0)
            row = int(ee[0])
            col = int(ee[1])
            print(row, col)
            print(ee, np.std(l, axis = 0))
            c_c = [col, row]
            for (x, y, w, h) in r_eyes:
                cv2.circle(image,(int(x+w/2), int(y+h/2)), 5, (0,255,0), -1)
                c_i = [int(x+w/2), int(y+h/2)]
                print ("Centroide image ", c_i, "Centroide cuadro", c_c, " ", w, h)
                if abs(c_i[1]-c_c[1])<=int(1.1*h) and abs(c_i[0]-c_c[0])<=int(1.1*w/2):
                    print("Exito")
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            # show the images
            cv2.circle(image,(col, row), 5, (0,255,0), -1)
            cv2.circle(output,(col, row), 5, (0,255,0), -1)
        cv2.imshow("images", np.hstack([image, output]))
        cv2.waitKey(0)

