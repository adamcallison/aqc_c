import aqc_c

import sys
sys.path.append("/Users/adam/Code/qrw3")
import util,hams
import pandas as pd
import numpy as np
import aqc
from time import time

m2s_esm=np.array([5.6503,7.0485,8.3781,9.7203,11.0517,12.3869,13.6843,15.0131,\
16.3593,17.615,18.904,20.2567,21.5778,22.8581,24.1733,25.4636])

def get_s(p, q):
    q = np.log(0.5)/np.log(q)
    return lambda x:_get_s_(p)(x**q)

def script():
    n=20
    msi=pd.read_csv('/Users/adam/Data/m2s/instances.csv',index_col='id')
    msin=msi.loc[msi.nqubits==n]
    inst=msin.iloc[0]
    Hp=util.m2sHp_from_id('/Users/adam/Data/m2s/instances/',inst.name)
    Hw=hams.zhypercube(nqubits=n)
    #Hw2=hams.hypercube(nqubits=n)
    Hpv=Hp.diagonal()
    g=m2s_esm[n-5]/2
    Hwv=Hw.diagonal()*g
    psi=np.ones(2**n,dtype=np.complex)/np.sqrt(2**n)
    sol=0
    Tmax=1
    s=np.arange(0,1.001,0.001)
    print('starting...')
    tb=time()
    prob = aqc_c.aqc(Hpv,Hwv,s,psi,Tmax,sol)
    ta=time()
    print('time taken: ',ta-tb)
    return s*Tmax,prob,psi
