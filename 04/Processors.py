import numpy as np 
from MathUtils import * 
from NEP import NEP, calculate_cutoff


def load_data():
    x_train = np.load('raw_data/x_unred_mergedth4501.npy')
    x_test = np.load('raw_data/x_unred_th_test_sim.npy')

    y_train = np.load('raw_data/y_mergedth4501.npy')
    y_test = np.load('raw_data/y_th_test_sim.npy')

    x_train = x_train[:10000]
    y_train = y_train[:10000]

    return x_train, y_train, x_test, y_test



def raw_to_pts(unred, factor, pts_type):
    Nper = 500000
    ntotal = unred.shape[0]
    n_loop = int(ntotal//Nper) + 3

    if(pts_type=='vertices'):
        n_pts = 4 
    elif(pts_type=='faces'):
        n_pts = 4 
    elif(pts_type=='edges'):
        n_pts =6     
    else:
        print('invalid pts_type')
        exit()

    pts8 = np.zeros((ntotal,n_pts*2,3),dtype=np.float32)    
    
    for i in range(n_loop):
        print('%d/%d'%(i,n_loop))
        if((i+1)*Nper < ntotal):
            pts8[i*Nper:(i+1)*Nper] = unred_to_8pts(unred[i*Nper:(i+1)*Nper],factor,pts_type)
        else:
            pts8[i*Nper:] = unred_to_8pts(unred[i*Nper:],factor,pts_type)
            break

    pts8 = pts8.astype(np.float32)
    return pts8 
        
def pts_to_nep(pts_tetra, hypers, nep_cutoff):

    n2_pts = pts_tetra.shape[1] 
    n_pts = n2_pts // 2 

    if(nep_cutoff==None):
        nep_cutoff = calculate_cutoff(pts_tetra)

    Nper = 50000

    nep = NEP(pts_tetra[:100],0)
    nep.set_hypers([int(hypers[0]),int(hypers[1]),int(hypers[2]),nep_cutoff])
    nep.set_npts(n_pts)
    g = nep.get_g_fast()

    ng = g.shape[1]
    lnp = len(pts_tetra)
    g_all = np.zeros((lnp,ng),dtype=np.float32)
    n_loop = int(lnp//Nper) + 3



    for i in range(n_loop):
        print('%d/%d'%(i,n_loop))

        if((i+1)*Nper < lnp):
            nep = NEP(pts_tetra[i*Nper:(i+1)*Nper],0)
            nep.set_hypers([int(hypers[0]),int(hypers[1]),int(hypers[2]),nep_cutoff])
            nep.set_npts(n_pts)
            gc = nep.get_g_fast()
            g_all[i*Nper:(i+1)*Nper] = gc
        else:
            nep = NEP(pts_tetra[i*Nper:],0)
            nep.set_hypers([int(hypers[0]),int(hypers[1]),int(hypers[2]),nep_cutoff])
            nep.set_npts(n_pts)
            gc = nep.get_g_fast()
            g_all[i*Nper:] = gc
            break

    g_all = g_all.astype(np.float32)
    return g_all, nep_cutoff    




























def dummy():
    pass 