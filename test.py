#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import PIL.Image
import pylab as pl

from poissonblending import blend 


img_mask = np.asarray(PIL.Image.open('./me_mask.png'))
img_mask = img_mask[:,:,:3] # remove alpha
img_source = np.asarray(PIL.Image.open('./me.png'))
img_source = img_source[:,:,:3] # remove alpha
img_target = np.asarray(PIL.Image.open('./target.png'))


nbsample=500
off = (35,-15)

img_ret1 = blend(img_target, img_source, img_mask, offset=off)
img_ret3 = blend(img_target, img_source, img_mask, reg=5,eta=1, nbsubsample=nbsample,offset=off,adapt='linear')
img_ret4 = blend(img_target, img_source, img_mask, reg=5,eta=1, nbsubsample=nbsample,offset=off,adapt='kernel')

#%%
fs=30
f, axarr = pl.subplots(1, 5,figsize=(30,7))
newax = f.add_axes([0.15, 0, 0.32, 0.32], anchor='NW', zorder=1)
newax.imshow(img_mask)
newax.axis('off')
newax.set_title('mask')
axarr[0].imshow(img_source)
axarr[0].set_title('Source', fontsize=fs)
axarr[0].axis('off')
axarr[1].imshow(img_target)
axarr[1].set_title('target', fontsize=fs)
axarr[1].axis('off')
axarr[2].imshow(img_ret1)
axarr[2].set_title('[Perez 03]', fontsize=fs)
axarr[2].axis('off')
axarr[3].imshow(img_ret3)
axarr[3].set_title('Linear', fontsize=fs)
axarr[3].axis('off')
axarr[4].imshow(img_ret4)
axarr[4].set_title('Kernel', fontsize=fs)
axarr[4].axis('off')
pl.subplots_adjust(wspace=0.1)
pl.tight_layout()
pl.show()

