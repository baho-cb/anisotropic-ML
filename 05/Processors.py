import numpy as np 
from PtsPlacement import PtsPlacement 
from NEP import NEP, calculate_cutoff


def load_data(debug_mode=False):
    x_train = np.load('raw_data/x_train.npy')
    x_test = np.load('raw_data/x_test.npy')

    y_train = np.load('raw_data/y_train.npy')
    y_test = np.load('raw_data/y_test.npy')

    yield_mask = np.abs(y_train)<15.0
    x_train = x_train[yield_mask]
    y_train = y_train[yield_mask]

    if(debug_mode):
        x_train = x_train[:10000]
        y_train = y_train[:10000]

    return x_train, y_train, x_test, y_test



def raw_to_pts(unred, hparams):
    Nper = 500000
    ntotal = unred.shape[0]
    n_loop = int(ntotal//Nper) + 3

    pts_placer = PtsPlacement(hparams)
    n_pts = pts_placer.npts

    allpts = np.zeros((ntotal,n_pts*2,3),dtype=np.float32)    
    
    for i in range(n_loop):
        print('%d/%d'%(i,n_loop))
        if((i+1)*Nper < ntotal):
            allpts[i*Nper:(i+1)*Nper] = pts_placer.place_pts_per_chunk(unred[i*Nper:(i+1)*Nper])
        else:
            allpts[i*Nper:] = pts_placer.place_pts_per_chunk(unred[i*Nper:])
            break

    allpts = allpts.astype(np.float32)
    return allpts 
        
def pts_to_nep(pts_all, hparams, device_id, nep_cutoff):

    nep_hypers = hparams['nep_hyper']
    n2_pts = pts_all.shape[1] 
    n_pts = n2_pts // 2 

    if(nep_cutoff==None):
        nep_cutoff = calculate_cutoff(pts_all)

    Nper = 50000

    nep = NEP(pts_all[:100])
    nep.set_hypers([int(nep_hypers[0]),int(nep_hypers[1]),int(nep_hypers[2]),nep_cutoff])
    nep.set_npts(n_pts)
    nep.set_device(device_id)
    nep.set_pts_types(hparams)

    g = nep.get_g_fast()

    ng = g.shape[1]
    lnp = len(pts_all)
    g_all = np.zeros((lnp,ng),dtype=np.float32)
    n_loop = int(lnp//Nper) + 3



    for i in range(n_loop):
        print('%d/%d'%(i,n_loop))

        if((i+1)*Nper < lnp):
            nep = NEP(pts_all[i*Nper:(i+1)*Nper])
            nep.set_hypers([int(nep_hypers[0]),int(nep_hypers[1]),int(nep_hypers[2]),nep_cutoff])
            nep.set_npts(n_pts)
            nep.set_device(device_id)
            nep.set_pts_types(hparams)
            gc = nep.get_g_fast()
            g_all[i*Nper:(i+1)*Nper] = gc
        else:
            nep = NEP(pts_all[i*Nper:])
            nep.set_hypers([int(nep_hypers[0]),int(nep_hypers[1]),int(nep_hypers[2]),nep_cutoff])
            nep.set_npts(n_pts)
            nep.set_device(device_id)
            nep.set_pts_types(hparams)
            gc = nep.get_g_fast()
            g_all[i*Nper:] = gc
            break
        
    return g_all, nep_cutoff    




























def dummy():
    pass 