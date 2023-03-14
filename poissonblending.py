#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
import pylab as pl
from scipy.spatial.distance import cdist

import sys
import ot


import sklearn.cluster as skcluster

def subsample(G_src,G_tgt,nb):
    ids=np.random.permutation(G_src.shape[0])
    Xs=G_src[ids[:nb],:]
    idt=np.random.permutation(G_tgt.shape[0])
    Xt=G_tgt[idt[:nb],:]

    return Xs,Xt


def getGradient_flatten(im):
    [grad_x,grad_y] =np.gradient(im.astype(float))
    return np.vstack((grad_x.flatten(),grad_y.flatten())).T

def adapt_Gradients_linear(G_src,G_tgt,mu=1e-2,eta=1e-6,nb=100,bias=True):
    Xs, Xt = subsample(G_src,G_tgt,nb)

    ot_mapping=ot.da.LinearTransport()
    ot_mapping.fit(Xs,Xt=Xt)
    return ot_mapping.transform(G_src)


def adapt_Gradients_kernel(G_src,G_tgt,mu=1e2,eta=1e-8,nb=10,bias=False,sigma=1e2):
    Xs, Xt = subsample(G_src,G_tgt,nb)

    ot_mapping_kernel=ot.da.MappingTransport(mu=mu,eta=eta,sigma=sigma,bias=bias, verbose=True)
    ot_mapping_kernel.fit(Xs,Xt=Xt)

    return ot_mapping_kernel.transform(G_src)


# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask

def blend(img_target, img_source, img_mask_raw, nbsubsample=100, offset=(0, 0),adapt='none',reg=1.,eta=1e-9,visu=0,verbose=False):
    # compute regions to be blended

    if verbose:
        print("Reticulating splines")
    region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0]-offset[0], img_source.shape[0]),
            min(img_target.shape[1]-offset[1], img_source.shape[1]))
    region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])
    #print region_size

    # clip and normalize mask image
    img_mask = img_mask_raw[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True



    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()

    # adapt_gradient

    G_src_tot = np.ndarray((region_size[0]*region_size[1],6),dtype=float)
    G_tgt_tot = np.ndarray((region_size[0]*region_size[1],6),dtype=float)


    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
        G_src_tot[:,2*num_layer:(2*num_layer+2)] = getGradient_flatten(s.astype(float))
        G_tgt_tot[:,2*num_layer:(2*num_layer+2)] = getGradient_flatten(t.astype(float))

    G_src = G_src_tot
    G_tgt = G_tgt_tot


    if verbose:
        print("Reticulating gradients")
    if adapt=='none':
        newG = G_src
    elif adapt=='linear':
        newG = adapt_Gradients_linear(G_src,G_tgt,mu=reg,eta=eta,nb=nbsubsample)
    elif adapt=='kernel':
        newG = adapt_Gradients_kernel(G_src,G_tgt,mu=reg,eta=eta,nb=nbsubsample)

    newGrad=newG

    # for each layer (ex. RGB)
    if verbose:
        print("Reticulating fish")
    im_return = img_target.copy()
    for num_layer in range(img_target.shape[2]):
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
        [grad_t_x,grad_t_y] =np.gradient(t.astype(float))
        t = t.flatten()

        grad_x = newGrad[:,2*num_layer].reshape(region_size[0],region_size[1])
        grad_y = newGrad[:,2*num_layer+1].reshape(region_size[0],region_size[1])

        g = np.zeros(img_mask.shape)
        g[1:, :] =grad_x[:-1, :] - grad_x[1:, :]
        g[:, 1:] = g[:, 1:] - grad_y[:, 1:]+grad_y[:, :-1]
        b=g.flatten()

        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        im_return[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    if verbose:
        print("Reticulating done")

    return im_return
