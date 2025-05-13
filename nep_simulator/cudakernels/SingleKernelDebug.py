import numpy as np 

# from NEP import NEP

def single_kernel_debug(pos,nep_cutoff,n_ang,lmax,nd):

    g_ang = np.zeros((n_ang+1)*lmax)

    meanx = 0.0 
    meany = 0.0
    meanz = 0.0

    for i in range(0,nd):
        meanx += pos[i,0]
        meany += pos[i,1]
        meanz += pos[i,2]
        
    meanx /= float(nd)
    meany /= float(nd)
    meanz /= float(nd)


    db = []
    for j in range(0,nd):
        r_ij_x = pos[j,0] - meanx
        r_ij_y = pos[j,1] - meany
        r_ij_z = pos[j,2] - meanz

        r_ij_norm = np.sqrt(r_ij_x*r_ij_x + r_ij_y*r_ij_y + r_ij_z*r_ij_z)
        r_ij_norm_cut = r_ij_norm/nep_cutoff

        x_ij = 2.0*(r_ij_norm_cut - 1.0)*(r_ij_norm_cut - 1.0) - 1.0
        fc_ij = 0.5*(1.0 + np.cos(np.pi*r_ij_norm_cut))

        

        for k in range(0,nd):
            r_ik_x = pos[k,0] - meanx
            r_ik_y = pos[k,1] - meany
            r_ik_z = pos[k,2] - meanz

            r_ik_norm = np.sqrt(r_ik_x*r_ik_x + r_ik_y*r_ik_y + r_ik_z*r_ik_z)
            r_ik_norm_cut = r_ik_norm/nep_cutoff

            x_ik = 2.0*(r_ik_norm_cut - 1.0)*(r_ik_norm_cut - 1.0) - 1.0
            fc_ik = 0.5*(1.0 + np.cos(np.pi*r_ik_norm_cut))

            chebo_ik_n_1 = 1.0
            chebo_ik_n = x_ik
            chebo_ik_next = 0.0
            chebo_ij_n_1 = 1.0
            chebo_ij_n = x_ij
            chebo_ij_next = 0.0
            # print(fc_ij)

            for i_cheb in range(0,n_ang+1):

                leg_n_1 = 1.0
                dot = (r_ij_x*r_ik_x + r_ij_y*r_ik_y + r_ij_z*r_ik_z)/(r_ij_norm*r_ik_norm)
                leg_n = dot
                leg_next = 0.0
                
                chebo_angular =  ((1.0 + chebo_ij_n_1)/2.0)*fc_ij*((1.0 + chebo_ik_n_1)/2.0)*fc_ik   
                if(i_cheb==0):
                    db.append(chebo_angular)

                g_ang[i_cheb*lmax + 0] += leg_n*chebo_angular
                for i_leg in range(1,lmax):
                    leg_next = ((2.0*i_leg + 1.0)*leg_n*dot - i_leg*leg_n_1)/(i_leg + 1.0)
                    leg_n_1 = leg_n
                    leg_n = leg_next
                    g_ang[i_cheb*lmax + i_leg] += leg_next*chebo_angular


                chebo_ik_next = chebo_ik_n*x_ik*2.0 - chebo_ik_n_1    
                chebo_ik_n_1 = chebo_ik_n
                chebo_ik_n = chebo_ik_next

                chebo_ij_next = chebo_ij_n*x_ij*2.0 - chebo_ij_n_1
                chebo_ij_n_1 = chebo_ij_n
                chebo_ij_n = chebo_ij_next
    # db.sort()
    # db = np.array(db)
    # print(db.shape)
    # print(db)
    return g_ang            


    


# pts = np.load('../pos_debug.npy')

# p0 = pts[1]
# g_ang1 = single_kernel_debug(p0,4.5,8,4,12)

# p0 = p0[np.newaxis]
# nep = NEP(p0,0)
# nep.set_hypers([10,8,4,4.5])
# nep.set_npts(6)
# nep.calculate_angular()
# g_ang2 = nep.g_ang[0]

# # print(g_ang1.shape)
# # print(g_an
# # g2.shape)
# # print(g_ang1)
# # print(g_ang2)



















