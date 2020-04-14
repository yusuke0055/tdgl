#磁気ドメインパターン形成TDGL方程式 多くのファイルを生成
#初期値の用意
import cProfile
import numpy as np
from matplotlib import pyplot as plt
import cv2
import copy
from scipy import signal
import pyfftw
import os
import time

def tdgl(hv,sv,seed,save=False):
    #初期値の用意
    class_name = str( int( round( sv*1e5 ))).zfill(7)
    if seed == "random":
        seed_name = "random"
        np.random.seed()
    else:
        seed_name = str(seed).zfill(4)    
       np.random.seed(seed)
#        np.random.seed()
        
    print(seed_name,class_name)

    classpath = "save/{}".format(class_name)
    path = "save/{}/{}".format(class_name,seed_name)
    #    path = "/home/yusuke/nas/lab_member_directories/2021_hamano/tdgl/{}/{}".format(class_name,seed_name)
    imgpath = "{}/img".format(path)
    npypath = "{}/npy".format(path)
    
    if not os.path.isdir(classpath):
        os.mkdir(classpath)
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(imgpath):
        os.mkdir(imgpath)
    if not os.path.isdir(npypath):
        os.mkdir(npypath)
        
    Phi_r_ini=np.ones([512,512],dtype=np.complex)
    Phi_r=Phi_r_ini+0.1*np.random.rand(512,512)
    Phi_k=np.zeros([512,512],dtype=np.complex)

    delta_t=0.1
    beta=4.0
    alpha=2.5
    gamma=2.0/np.pi

    #k=(k_x,k_y)の用意
    k_x=np.linspace(0,511,512)
    for i in range(len(k_x)):
        if i>len(k_x)//2:
            k_x[i]=k_x[i]-len(k_x)
        #else:
        #    k_x[i]=k_x[i

    k_y=copy.copy(k_x)
    tmp = np.zeros([512, 512])
    for i in range(512):
        for j in range(512):
            tmp[j,i]=k_x[i]*k_x[i]+k_y[j]*k_y[j]
    kk_sqrt = 2.0*np.pi*np.sqrt(tmp) / len(k_x)
    keisu = 1 / (1+tmp*delta_t*beta*4.0*np.pi**2/len(k_x)**2)
    
    Lambda_ini=np.ones([512,512])
    myu_0=0.3

    Lambda=Lambda_ini+0.25*np.random.normal(0, myu_0, (512,512))
    Lambda_FFT=np.fft.fft2(Lambda)

    a0=4.0
    a1=2.0*np.pi

    G_k=np.zeros([512,512])
    G_k=a0-a1*kk_sqrt
    
    h_ini_r=1.4
    h=np.ones([512,512])
    h=h_ini_r*h
    
    H_k=np.fft.fft2(h)
    v=np.ones([512,512])
    h_v=v*hv

    #JPSJ(2005)013002.(7)式の用意:時刻t=0のもの
    Phi_r3=Phi_r[:]**3
    First=Lambda*(Phi_r[:]-Phi_r3)
    #First=(Phi_r[0,:,:]-Phi_r3)

    First_FFT=pyfftw.interfaces.numpy_fft.fft2(First)
    print('First', np.mean(First), 'First_FFT', np.mean(First_FFT))
    Phi_k[:]=pyfftw.interfaces.numpy_fft.fft2(Phi_r[:])

    mask_area = np.ones((512, 512))
    for l in range(-256, 256):
        for k in range(-256, 256):
            if k**2+l**2 > 170**2:
                mask_area[k, l] = 0
    mask_area = mask_area == 0
    First_FFT[mask_area] = 0.0 + 0.0j
    #時刻t+ΔtのPhi_k k=(k_x,k_y)とk^2=k_x^2+k_y^2これの平方根sを用意する
    
    STEPS = int( 2.0*np.round( 1.4/(h_v[0,0]*delta_t) ))
#    STEPS = 1
    Phi_r_list = []
    Phi_k_list = []
    
    for i in range(STEPS):
        #print("{:7}/{:7}".format(i,STEPS),end="\r")
        #Phi_r_list.append(copy.copy(Phi_r[:]))
        #Phi_k_list.append(copy.copy(Phi_k[:]))    

        Phi_k[:] = keisu*(delta_t*(alpha*First_FFT+H_k-gamma*G_k*Phi_k[:])+Phi_k[:])
        Phi_r[:] = pyfftw.interfaces.numpy_fft.ifft2(Phi_k[:], overwrite_input=True)

        Phi_r[511,1:510]=Phi_r[0,1:510]
        Phi_r[1:510,511]=Phi_r[1:510,0]
        Phi_r[511,511]=Phi_r[0,0]
        Phi_r[0,511]=Phi_r[0,0]
        Phi_r[511,0]=Phi_r[0,0]

        Phi_r3 = Phi_r ** 3
        First=Lambda*(Phi_r[:]-Phi_r3)
        First_FFT=pyfftw.interfaces.numpy_fft.fft2(First, overwrite_input=True)
        First_FFT[mask_area]=0.0+0.0j # 先程計算した範囲で一気に0にする

        h_temp=h-delta_t*h_v
        if h_temp[0,0] <= 1.0e-5:
            h=h_temp*0.0
        else:
            h=h_temp
            
        H_k=pyfftw.interfaces.numpy_fft.fft2(h, overwrite_input=True) #need
        
        #saving
        if save and (i%10 == 0):
            iind = "{}".format(i).zfill(9)
            np.save("{}/{}".format(npypath,iind) ,[h[0,0],Phi_r])
            img = Phi_r.real
            im_bin = (img>0)*255
            im_bin_inv = cv2.bitwise_not(im_bin) + 256
            cv2.imwrite("{}/{}.png".format(imgpath,iind),im_bin_inv)        
