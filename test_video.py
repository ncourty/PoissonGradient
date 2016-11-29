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


img_mask = np.asarray(PIL.Image.open('./data/joconde_0_mask.jpg'))
img_mask = img_mask[:,:,:3] # remove alpha
img_mask.flags.writeable = True

#img_source = np.asarray(PIL.Image.open('./testimages/me_flipped.png'))
#img_source = img_source[:,:,:3]
#img_source.flags.writeable = True

img_target = np.asarray(PIL.Image.open('./data/joconde_0.jpg'))
img_target.flags.writeable = True


#img_mask2=cv2.imread('testimages/mask_video.png')
#img_mask=img_mask2>0
mask2= np.maximum(cv2.Laplacian(img_mask,cv2.CV_64F),0)>20

# nico : change the resolution of the video to match mask resolution
cap.set(3,img_mask.shape[1])
cap.set(4,img_mask.shape[0])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame=frame[:,::-1,:]

    # Our operations on the frame come here
    frame_mask=(img_mask>0)*frame+(img_mask==0)*cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    cv2.imshow('frame',frame_mask)
    if (cv2.waitKey(1) & 0xFF) in [ ord(' '),ord('q')]:
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2RGB)
#frame_mask2= cv2.cvtColor((img_mask2>0)*frame, cv2.COLOR_BGR2RGB)
I = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#pl.figure(1)
#pl.subplot(211)
#pl.imshow(I)
#pl.subplot(212)
#pl.imshow(frame_mask2)
#
#pl.show()
nbsample=500
off = (0,0)
img_ret4 = blend(img_target, I, img_mask, reg=5, nbsubsample=nbsample,offset=off,adapt='kernel')

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
