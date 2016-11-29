# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 08:24:30 2016

@author: rflamary
"""

import numpy as np
import cv2
import pylab as pl

cap = cv2.VideoCapture(0)

import numpy as np
import PIL.Image


from poissonblending import blend

fname_mask='./data/joconde_0_mask.jpg'
fname_img='./data/joconde_0.jpg'

mask_list=['./data/joconde_0_mask.jpg',
           './data/monet_portrait_mask.png']

img_list=['./data/joconde_0.jpg',
           './data/monet_portrait.png']

nimg=len(img_list)
idimg=0


def load_images(fname_mask,fname_img):

    img_mask = np.asarray(PIL.Image.open(fname_mask))

    if len(img_mask.shape)<3:
        img_mask = np.tile(img_mask[:,:,None],(1,1,3)) # remove alpha
    img_mask = img_mask[:,:,:3]
    img_mask.flags.writeable = True

    #img_source = np.asarray(PIL.Image.open('./testimages/me_flipped.png'))
    #img_source = img_source[:,:,:3]
    #img_source.flags.writeable = True

    img_target = np.asarray(PIL.Image.open(fname_img))
    img_target = img_target[:,:,:3]
    img_target.flags.writeable = True


    #img_mask2=cv2.imread('testimages/mask_video.png')
    #img_mask=img_mask2>0
    mask2= np.maximum(cv2.Laplacian(img_mask,cv2.CV_64F),0)>20

    return img_mask,img_target,mask2


img_mask,img_target,mask2=load_images(fname_mask,fname_img)
# nico : change the resolution of the video to match mask resolution
cap.set(3,img_mask.shape[1])
cap.set(4,img_mask.shape[0])

etape=0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame=frame[:,::-1,:]

    # Our operations on the frame come here
    if etape==0:
        frame_mask=cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
    elif etape==1:
        frame_mask=(img_mask>0)*frame+(img_mask==0)*cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)

    k=cv2.waitKey(1) & 0xFF
    if k in [ ord(' ')]:
        if etape==1:
            doit=True
            break
    if k in [ ord('q')]:
        doit=False
        break
    if k in [ord('v')]:
         etape=1
    if k in [ord('i')]:
         idimg=(idimg+1) % nimg
         fname_mask=mask_list[idimg]
         fname_img=img_list[idimg]
         img_mask,img_target,mask2=load_images(fname_mask,fname_img)
    # Display the resultinfname_maskg frame
    cv2.imshow('frame',frame_mask)



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


I = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if doit:
    nbsample=500
    off = (0,0)
    img_ret4 = blend(img_target, I, img_mask, reg=5, nbsubsample=nbsample,offset=off,adapt='kernel',verbose=True)

    #%%
    fs=30
    f, axarr = pl.subplots(1, 3,figsize=(30,10))
    newax = f.add_axes([0.15, 0, 0.32, 0.32], anchor='NW', zorder=1)
    newax.imshow(img_mask)
    newax.axis('off')
    newax.set_title('mask')
    axarr[0].imshow(I)
    axarr[0].set_title('Source', fontsize=fs)
    axarr[0].axis('off')
    axarr[1].imshow(img_target)
    axarr[1].set_title('target', fontsize=fs)
    axarr[1].axis('off')
    axarr[2].imshow(img_ret4)
    axarr[2].set_title('Kernel', fontsize=fs)
    axarr[2].axis('off')
    pl.subplots_adjust(wspace=0.1)
    pl.tight_layout()

    pl.savefig('./output/webcam.png')
    pl.show()
