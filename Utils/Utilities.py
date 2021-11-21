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
from skimage.transform import rotate
import networkx as nx
from skimage import measure
from skimage import color
import copy

def toLCh(img):
    lab=color.rgb2lab(img)
    L=lab[:,:,0]
    a=lab[:,:,1]
    b=lab[:,:,2]
    C=(a**2+b**2)**0.5
    h=np.arctan2(b,a)
    Lch=np.concatenate((L.reshape(L.shape[0],L.shape[1],1),
                        C.reshape(L.shape[0],L.shape[1],1),
                        h.reshape(L.shape[0],L.shape[1],1)),axis=2)
    return Lch

def maxmod(p,th=5000):
  for i in range(len(p[0][:])):
    if p[0][-i]>th:
      max=p[1][-i]
      break
  return max

def cmask(mask):
  sel=0
  for i in mask.ravel():
    if i:
      sel=sel+1
  return sel

def ucut(img,f):
  cx=int(img.shape[0]/(f*2))
  cy=int(img.shape[1]/(f*2))
  return img[cx:(img.shape[0]-cx),cy:(img.shape[1]-cy)]

def Assem(img,cimg,f):
  m=np.full(img.shape, False)
  cx=int(img.shape[0]/(f*2))
  cy=int(img.shape[1]/(f*2))
  m[cx:img.shape[0]-cx,cy:img.shape[1]-cy]=cimg
  return m

def get_Val(img):
  return img[:, :, 2]

def prePro(img):
  hsv_i= rgb2hsv(img)
  hsv_i = cv2.blur(hsv_i,(25,25),0)
  return get_Val(hsv_i)

def get_binary(img,th=0.8):
  return img > th

def getBiggestCont(binaryImg,n=3):
  contours = measure.find_contours(binaryImg, 0.8)
  contours.sort(key=lambda x: x.shape[0], reverse=True)
  return contours[0:n]

def is_In(In,Ou):
  if ((In[0,0]>Ou[:,0].min())and(In[0,0]<Ou[:,0].max())and(In[0,1]>Ou[:,1].min())and(In[0,1]<Ou[:,1].max())):
    return True
  else:
    return False

def is_Con(Contour,img):
  if((Contour[:,0].min()==0)or(Contour[:,1].min()==0)or(Contour[:,0].max()==(img.shape[0]-1))or(Contour[:,0].max()==(img.shape[1]-1))):
    return False
  else:
    return True

def closedCon(contours,binary_img):
  temp=[]
  for c in contours:
    if is_Con(c,binary_img):
      temp.append(c)
  return temp

def contenedContours(contours):
  cont=[]
  for ic in contours:
    for cc in contours:
      if np.array_equal(ic,cc):
        break
      if is_In(ic,cc):
        cont.append(ic)
        break
  return cont

def nonContened(contours,contened):
  conte=contours
  for c in contened:
    conte.remove(c)
  return conte

def containerContours(Noncontent,Content):
  temp=[]
  for cc in Noncontent:
    for ic in Content:
      if is_In(ic,cc):
        temp.append(cc)
        break
  return temp

def similarForms(contours,pcaModel):
  Meta=[]
  for c in contours:
    PC = pcaModel.fit_transform(c)
    Cc = np.expand_dims(PC.astype(np.float32), 1)
    Cc = cv2.UMat(Cc)
    areaR = cv2.contourArea(Cc)
    areaBB=(PC[:,0].max()-PC[:,0].min())*(PC[:,1].max()-PC[:,1].min())
    Meta.append((c,(areaR/areaBB)))
  Meta.sort(key=lambda x: x[1], reverse=True)
  return Meta[0][0]

def cont2Img(img,Cont):
  stc=np.full(img.shape, False)
  stc[Cont.astype('int')[:,0],Cont.astype('int')[:,1]]=True
  return stc

def applyMask(img,binary):
  con3=np.zeros(img.shape)
  con3[:,:,0]=binary
  con3[:,:,1]=binary
  con3[:,:,2]=binary
  ROI1=np.int_(img)*con3
  return np.uint8(ROI1)

def stickerDetection(img,th=0.8):
  val_img=prePro(img)
  binary_img = get_binary(val_img,th)
  contours=getBiggestCont(binary_img,n=10)
  contours=closedCon(contours,binary_img)
  cont=contenedContours(contours)
  conte=nonContened(contours,cont)
  contours=containerContours(conte,cont)
  pca = PCA(n_components=2)
  selected_forms=similarForms(contours,pca)
  stc=cont2Img(binary_img,selected_forms)
  con=convex_hull_image(stc)
  return applyMask(img,con)

def stickerFilter(img,th=0.8):
  val_img=prePro(img)
  binary_img = get_binary(val_img,th)
  contours=getBiggestCont(binary_img,n=10)
  contours=closedCon(contours,binary_img)
  cont=contenedContours(contours)
  conte=nonContened(contours,cont)
  contours=containerContours(conte,cont)
  pca = PCA(n_components=2)
  selected_forms=similarForms(contours,pca)
  stc=cont2Img(binary_img,selected_forms)
  con=convex_hull_image(stc)
  return applyMask(img,np.invert(con))

def stickerCut(img,th=0.8):
  val_img=prePro(img)
  binary_img = get_binary(val_img,th)
  contours=getBiggestCont(binary_img,n=10)
  contours=closedCon(contours,binary_img)
  cont=contenedContours(contours)
  conte=nonContened(contours,cont)
  contours=containerContours(conte,cont)
  pca = PCA(n_components=2)
  selected_forms=similarForms(contours,pca)
  stc=cont2Img(binary_img,selected_forms)
  con=convex_hull_image(stc)
  pth=300
  Recorte_aumento=500
  TIndex=np.where(con==True)
  Xmax=np.max(TIndex[0])+Recorte_aumento
  Xmin=np.min(TIndex[0])-Recorte_aumento
  Ymax=np.max(TIndex[1])+Recorte_aumento
  Ymin=np.min(TIndex[1])-Recorte_aumento
  if Xmin<pth:
    Xmin=pth
  if Ymin<pth:
    Ymin=pth
  if Xmax>con.shape[0]-pth:
    Xmax=con.shape[0]-pth
  if Ymax>con.shape[1]-pth:
    Ymax=con.shape[1]-pth
  con[:,:]=False
  con[Xmin:Xmax,Ymin:Ymax]=True
  return applyMask(img,con)

def get_X_U(img,mask,n_segments=800):
    lum = color.rgb2gray(img)
    mask1=lum>0

    m_slic = slic(img, n_segments=n_segments,sigma=5,slic_zero=True,mask=mask)

    RID=set(m_slic.flatten())
    f=np.zeros((img.shape[0],img.shape[1],4))
    f[:,:,0:3]=img[:,:,0:3]
    f[:,:,3]=m_slic

    DIRID={int(i):{'U':np.zeros((3)),'X':np.zeros((3))} for i in RID}
    indx=np.where(f[:,:,3]==1)
    f[indx[0],indx[1],:]

    for i in RID:
        indx=np.where(f[:,:,3]==i)
        x=np.mean(f[indx[0],indx[1],:],axis=0)
        u=np.std(f[indx[0],indx[1],:],axis=0)
        DIRID[i]['X']=x
        DIRID[i]['U']=u
    return DIRID


def get_SD_by_format(f,ind1,ind2,j):
    srd=f[ind1,ind2,j.reshape(-1,1)]
    x=np.mean(srd,axis=1)
    u=np.std(srd,axis=1)
    perc=np.percentile(srd,np.array([0,25,50,75,100]),axis=1)
    #hist=np.vectorize(pyfunc=(lambda t:np.histogram(f[ind1,ind2,t],bins=50,range=(0,255))),signature='()->(j),(k)')(j)
    hist=np.vectorize(pyfunc=(lambda dt,t:np.histogram(dt[t,:],bins=50,range=(0,1))),signature='(a,b),()->(j),(k)')(srd,j%3)
    Mo=np.vectorize(pyfunc=(lambda x,y:y[np.where(x==np.max(x))[0][0]]),signature='(j),(k)->()')(hist[0],hist[1])
    return np.vstack((x,u,perc,Mo))
v_get_SD_by_format=np.vectorize(pyfunc=get_SD_by_format,signature='(x,y,z),(a),(b),(j)->(q,w)')

def get_NSD_by_format(f,ind1,ind2,j,fac=1):
    srd=f[ind1,ind2,j.reshape(-1,1)]
    x=np.mean(srd,axis=1)
    u=np.std(srd,axis=1)
    nsrd=(srd-x.reshape(-1,1))/u.reshape(-1,1)
    srd=srd[:,np.prod(u.reshape(-1,1)*fac>(np.abs(nsrd)),axis=0).astype("bool")]
    x=np.mean(srd,axis=1)
    u=np.std(srd,axis=1)
    perc=np.percentile(srd,np.array([0,25,50,75,100]),axis=1)
    hist=np.vectorize(pyfunc=(lambda dt,t:np.histogram(dt[t,:],bins=50,range=(0,255))),signature='(a,b),()->(j),(k)')(srd,j%3)
    Mo=np.vectorize(pyfunc=(lambda x,y:y[np.where(x==np.max(x))[0][0]]),signature='(j),(k)->()')(hist[0],hist[1])
    return np.vstack((x,u,perc,Mo))
v_get_NSD_by_format=np.vectorize(pyfunc=get_SD_by_format,signature='(x,y,z),(a),(b),(j)->(q,w)')

def pack_segments(DIRID,f,i):
    indx=np.where(f[:,:,12]==i)
    SD_rgb_hsv_lab=v_get_SD_by_format(f,indx[0],indx[1],np.arange(0,12).reshape(-1,3))
    DIRID[i]['rgb_mean']=SD_rgb_hsv_lab[0,0]
    DIRID[i]['rgb_std']=SD_rgb_hsv_lab[0,1]
    DIRID[i]['rgb_per0']=SD_rgb_hsv_lab[0,2]
    DIRID[i]['rgb_per25']=SD_rgb_hsv_lab[0,3]
    DIRID[i]['rgb_per50']=SD_rgb_hsv_lab[0,4]
    DIRID[i]['rgb_per75']=SD_rgb_hsv_lab[0,5]
    DIRID[i]['rgb_per100']=SD_rgb_hsv_lab[0,6]
    DIRID[i]['rgb_mo']=SD_rgb_hsv_lab[0,7]
    DIRID[i]['hsv_mean']=SD_rgb_hsv_lab[1,0]
    DIRID[i]['hsv_std']=SD_rgb_hsv_lab[1,1]
    DIRID[i]['hsv_per0']=SD_rgb_hsv_lab[1,2]
    DIRID[i]['hsv_per25']=SD_rgb_hsv_lab[1,3]
    DIRID[i]['hsv_per50']=SD_rgb_hsv_lab[1,4]
    DIRID[i]['hsv_per75']=SD_rgb_hsv_lab[1,5]
    DIRID[i]['hsv_per100']=SD_rgb_hsv_lab[1,6]
    DIRID[i]['hsv_mo']=SD_rgb_hsv_lab[1,7]
    DIRID[i]['lab_mean']=SD_rgb_hsv_lab[2,0]
    DIRID[i]['lab_std']=SD_rgb_hsv_lab[2,1]
    DIRID[i]['lab_per0']=SD_rgb_hsv_lab[2,2]
    DIRID[i]['lab_per25']=SD_rgb_hsv_lab[2,3]
    DIRID[i]['lab_per50']=SD_rgb_hsv_lab[2,4]
    DIRID[i]['lab_per75']=SD_rgb_hsv_lab[2,5]
    DIRID[i]['lab_per100']=SD_rgb_hsv_lab[2,6]
    DIRID[i]['lab_mo']=SD_rgb_hsv_lab[2,7]
    DIRID[i]['lCh_mean']=SD_rgb_hsv_lab[3,0]
    DIRID[i]['lCh_std']=SD_rgb_hsv_lab[3,1]
    DIRID[i]['lCh_per0']=SD_rgb_hsv_lab[3,2]
    DIRID[i]['lCh_per25']=SD_rgb_hsv_lab[3,3]
    DIRID[i]['lCh_per50']=SD_rgb_hsv_lab[3,4]
    DIRID[i]['lCh_per75']=SD_rgb_hsv_lab[3,5]
    DIRID[i]['lCh_per100']=SD_rgb_hsv_lab[3,6]
    DIRID[i]['lCh_mo']=SD_rgb_hsv_lab[3,7]
    #Descriptores espaciales--------------------------------------------
    DIRID[i]['x_mean']=np.mean(indx[0])
    DIRID[i]['x_std']=np.std(indx[0])
    DIRID[i]['y_mean']=np.mean(indx[1])
    DIRID[i]['y_std']=np.std(indx[1])
    DIRID[i]['N']=indx[0].shape[0]
    return 0
v_pack_segments=np.vectorize(pyfunc=pack_segments,signature="(),(x,y,z),()->()")

def st_adjust(img):
    im=stickerDetection(img)

    st_mask=color.rgb2lab(im)[:,:,0]>80

    st_mask_sh=np.concatenate((st_mask.reshape(st_mask.shape[0],st_mask.shape[1],1),
                st_mask.reshape(st_mask.shape[0],st_mask.shape[1],1),
                st_mask.reshape(st_mask.shape[0],st_mask.shape[1],1)),axis=2)

    imm=st_mask_sh*im
    
    MB=np.mean(im[st_mask,:])

    adj_img=np.concatenate((img[:,:,0].reshape(img.shape[0],img.shape[1],1)*(200/MB),
                        img[:,:,1].reshape(img.shape[0],img.shape[1],1)*(200/MB),
                        img[:,:,2].reshape(img.shape[0],img.shape[1],1)*(200/MB)),axis=2).astype(int)
    
    return adj_img

def get_Statistical_Descriptors(img,mask,n_segments=800):
    lum = np.mean(mask,axis=2).astype(int)
    mask1=lum>0

    m_slic = slic(img, n_segments=n_segments,sigma=5,slic_zero=True,mask=mask1)
    
    RID=np.unique(m_slic.flatten())
    f=np.zeros((img.shape[0],img.shape[1],3+3+3+1))
    f[:,:,0:3]=img[:,:,0:3]
    f[:,:,3:6]=color.rgb2hsv(img)[:,:,0:3]
    f[:,:,6:9]=color.rgb2lab(img)[:,:,0:3]
    f[:,:,9]=m_slic

    DIRID={int(i):{'rgb_mean':np.zeros((1)),'rgb_std':np.zeros((1)),'rgb_per':np.zeros((1)),'rgb_mo':np.zeros((1)),
                   'lab_mean':np.zeros((1)),'lab_std':np.zeros((1)),'lab_per':np.zeros((1)),'lab_mo':np.zeros((1)),
                   'hsv_mean':np.zeros((1)),'hsv_std':np.zeros((1)),'hsv_per':np.zeros((1)),'hsv_mo':np.zeros((1)),
                  } for i in RID}

    v_pack_segments(DIRID,f,RID)
    return DIRID

def get_Normalized_Statistical_Descriptors_(img,mask,n_segments=800,angle=0):
    lum = np.mean(mask,axis=2).astype(int)
    mask1=lum>0

    img=rotate(img,angle)
    mask1=rotate(mask1,angle)

    m_slic = slic(image=img, n_segments=n_segments,sigma=5,slic_zero=True,mask=mask1)
    
    RID=np.unique(m_slic.flatten())
    f=np.zeros((img.shape[0],img.shape[1],3+3+3+3+1))
    f[:,:,0:3]=img[:,:,0:3]
    f[:,:,3:6]=color.rgb2hsv(img)[:,:,0:3]
    f[:,:,6:9]=color.rgb2lab(img)[:,:,0:3]
    LCh=toLCh(img)
    f[:,:,9:12]=LCh[:,:,0:3]
    f[:,:,12]=m_slic

    DIRID={int(i):{'rgb_mean':np.zeros((3)),'rgb_std':np.zeros((3)),'rgb_per':np.zeros((3)),'rgb_mo':np.zeros((3)),
                   'lab_mean':np.zeros((3)),'lab_std':np.zeros((3)),'lab_per':np.zeros((3)),'lab_mo':np.zeros((3)),
                   'hsv_mean':np.zeros((3)),'hsv_std':np.zeros((3)),'hsv_per':np.zeros((3)),'hsv_mo':np.zeros((3)),
                   'LCh_mean':np.zeros((3)),'LCh_std':np.zeros((3)),'LCh_per':np.zeros((3)),'LCh_mo':np.zeros((3))
                  } for i in RID}

    v_pack_segments(DIRID,f,RID)
    return DIRID,m_slic

def get_Statistical_Descriptors_(img,mask,n_segments=800,angle=0):
    lum = np.mean(mask,axis=2).astype(int)
    mask1=lum>0

    img=rotate(img,angle)
    mask1=rotate(mask1,angle)

    m_slic = slic(image=img, n_segments=n_segments,sigma=5,slic_zero=True,mask=mask1)
    
    RID=np.unique(m_slic.flatten())
    f=np.zeros((img.shape[0],img.shape[1],3+3+3+3+1))
    f[:,:,0:3]=img[:,:,0:3]
    f[:,:,3:6]=color.rgb2hsv(img)[:,:,0:3]
    f[:,:,6:9]=color.rgb2lab(img)[:,:,0:3]
    LCh=toLCh(img)
    f[:,:,9:12]=LCh[:,:,0:3]
    f[:,:,12]=m_slic
    DIRID={int(i):{'rgb_mean':np.zeros((1)),'rgb_std':np.zeros((1)),'rgb_mo':np.zeros((1)),
                   'lab_mean':np.zeros((1)),'lab_std':np.zeros((1)),'lab_mo':np.zeros((1)),
                   'hsv_mean':np.zeros((1)),'hsv_std':np.zeros((1)),'hsv_mo':np.zeros((1)),
                  } for i in RID}

    v_pack_segments(DIRID,f,RID)
    return DIRID,m_slic

def get_Normalized_Statistical_Descriptors_(img,mask,n_segments=800,angle=0):
    lum = np.mean(mask,axis=2).astype(int)
    mask1=lum>0

    wth=200
    fmw=np.prod(img>wth,axis=2)
    wth=100
    fmn=np.prod(img<wth,axis=2)
    fm=np.logical_or(fmn,fmw)
    img=st_adjust(img)

    img=rotate(img,angle,resize=True)
    mask1=rotate(mask1,angle,resize=True)
    fm=rotate(fm,angle,resize=True)

    m_slic = slic(image=img, n_segments=n_segments,sigma=5,slic_zero=True,mask=mask1)


    RID=np.unique(m_slic.flatten())
    tm_slic=copy.deepcopy(m_slic)


    tm_slic[np.where(fm)]=n_segments+5

    f=np.zeros((img.shape[0],img.shape[1],3+3+3+3+1))
    f[:,:,0:3]=(img[:,:,0:3]+np.array([0,0,0]))/np.array([1,1,1])
    f[:,:,3:6]=(color.rgb2hsv(img)[:,:,0:3]+np.array([0,0,0]))/np.array([1,1,1])
    f[:,:,6:9]=(color.rgb2lab(img)[:,:,0:3]+np.array([0,128,128]))/np.array([100,256,256])
    LCh=toLCh(img)
    f[:,:,9:12]=(LCh[:,:,0:3]+np.array([0,0,np.pi]))/np.array([100,(128*(2**0.5)),2*np.pi])
    f[:,:,12]=tm_slic

    DIRID={int(i):{'rgb_mean':np.zeros((1)),'rgb_std':np.zeros((1)),'rgb_mo':np.zeros((1)),
                   'lab_mean':np.zeros((1)),'lab_std':np.zeros((1)),'lab_mo':np.zeros((1)),
                   'hsv_mean':np.zeros((1)),'hsv_std':np.zeros((1)),'hsv_mo':np.zeros((1)),
                  } for i in RID}

    v_pack_segments(DIRID,f,RID)
    return DIRID,m_slic


def replace_err(data):
    return np.concatenate((data[:2],np.array(['c'+data[2][9:]])),axis=0)
v_replace_err=np.vectorize(replace_err,signature="(n)->(m)")
def no_spaces(data):
    f=np.array(data[0].split(" ")[1:]+data[1].split(" ")[1:])
    return np.array([f])
v_no_spaces=np.vectorize(pyfunc=no_spaces,signature="(m)->(k)")

def assemble_mask(xywh,img,ROI):
    marco=xywh
    mask=np.full((img.shape),0)
    mask[marco[1]:(marco[1]+marco[3]),marco[0]:(marco[0]+marco[2]),:]=ROI
    return mask

def get_graph_from_image(image,mask,desired_nodes=20,angle=0,normalized=False):
    if normalized:
      SD,segments=get_Normalized_Statistical_Descriptors_(image,mask,n_segments=desired_nodes,angle=angle)
    else:
      SD,segments=get_Statistical_Descriptors_(image,mask,n_segments=desired_nodes,angle=angle)
    nodes=np.array(list(SD))[:]
    node_features=np.vectorize(lambda SD,node:SD[node])(SD,nodes)
    G = nx.Graph()
    for node in nodes[1:]:
        data=np.array(list(node_features[node].items()))[:,1]
        #print(data.shape)
        #print(data)
        afeatures=np.concatenate((np.concatenate(data[:32]),data[32:]))
        #print(afeatures)
        n_features=afeatures.shape[0]
        G.add_node(node-1, features = afeatures)
    
    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    bneighbors=np.delete(bneighbors,np.where(bneighbors[1,:]==0),axis=1)
    bneighbors=np.delete(bneighbors,np.where(bneighbors[0,:]==0),axis=1)-1
    
    for i in range(bneighbors.shape[1]):
        if (bneighbors[0,i] != bneighbors[1,i]):
            G.add_edge(bneighbors[0,i],bneighbors[1,i])
    
    for node in nodes[1:]:
        G.add_edge(node-1,node-1)
    
    n = len(G.nodes)
    m = len(G.edges)
    h = np.zeros([n,n_features])
    edges = np.zeros([2*m,2])
    for e,(s,t) in enumerate(G.edges):
        edges[e,0] = s
        edges[e,1] = t
        
        edges[m+e,0] = t
        edges[m+e,1] = s
    for i in G.nodes:
        #print(G.nodes[i]["features"])
        h[i,:] = G.nodes[i]["features"]
    return SD,G, h, edges

def sample_central(SD,G,num=5,maxdeg=3):
    centers=np.vectorize(pyfunc=lambda i,SD: np.array([SD[i]["x_mean"],SD[i]["y_mean"]]),
             signature="(),()->(j)")(np.arange(1,len(G.nodes)),SD)
    c_node=np.argmin(np.linalg.norm(centers-np.mean(centers,axis=0),axis=1))
    sampled=[c_node]
    deg=1
    th_deg_nei=np.array(list(nx.single_source_shortest_path_length(G, c_node, cutoff=maxdeg).items()))
    #selected=int(centers.shape[0]*samp_frac)
    selected=num
    while len(sampled)!=selected:
        nd=th_deg_nei[np.where(th_deg_nei[:,1]==1)[0]][:,0]
        if (selected-len(sampled))<=len(nd):
            new=np.random.choice(nd,size=selected-len(sampled),replace=False)
        else:
            new=nd
        sampled=sampled+new.tolist()
        deg=deg+1
    return sampled