#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


# function to compute the projected probablities
# This is an implementation of Algorithm 1 in the Distributional perpective paper: 
# arXiv:1707.06887 [cs.LG]
def projected_prob(vmin, vmax, N, reward, discount, target_prob):
    delta = (vmax - vmin)/(N-1)
    z = np.array([ vmin + i*delta for i in range(N)])
    Tz = np.clip(reward + discount*z, vmin, vmax)
    b = (Tz-vmin)/delta
    l = np.floor(b).astype(int)
    small_shift = 1e-5
    u = np.ceil(b+small_shift).astype(int)
    projected_probs = np.zeros(N)
    for ii, lu in enumerate(zip(l,u)):
        ll, uu = lu
        if ll in range(N):
            projected_probs[ll]+=target_prob[ii]*(uu-b[ii])
        if uu in range(N):
            projected_probs[uu]+=target_prob[ii]*(b[ii]-ll)
    return projected_probs     

