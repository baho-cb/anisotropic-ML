import numpy as np 

Np = 420 
nangp1 = 9 
lmax = 4 
Nd = 12 

blocks = np.arange(Np*nangp1*lmax, dtype=np.int32)
threads = np.arange(Nd*Nd*6, dtype=np.int32)

for blx in blocks:
    for thx in threads: 

        index_grad_ij = (blx//lmax) * Nd  + (thx // 6)%Nd
        index_grad_ik = (blx//lmax) * Nd  + (thx // 6)//Nd

        # index_dgdteta_i = (blx//lmax) * Nd * 6 + (thx // Nd)
        # index_dgdteta_j = (blx//lmax) * Nd * 6 + (thx % Nd)*6
        # index_dgdteta_j = (blx//lmax) * Nd * 6 + (thx % Nd)


        index_target = blx*Nd*Nd*6 + thx    

        multi_index_grad_i = np.unravel_index(index_grad_ij, (Np, nangp1, Nd))
        multi_index_grad_j = np.unravel_index(index_grad_ik, (Np, nangp1, Nd))
        multi_index_target = np.unravel_index(index_target, (Np, nangp1*lmax, Nd*Nd,6))

        # multi_index_dgdteta_i = np.unravel_index(index_dgdteta_i, (Np, nangp1, Nd,6))
        # multi_index_dgdteta_j = np.unravel_index(index_dgdteta_j, (Np, nangp1, Nd,6))

        print(multi_index_grad_i,multi_index_grad_j,multi_index_target)
        # print(multi_index_dgdteta_i,multi_index_dgdteta_j)



        # multi_index_target = np.unravel_index(index_target, (Np, nangp1*lmax, Nd*Nd, 6))

        # if(multi_index_target==(2,14,45,3)):

        #     print(multi_index_grad,multi_index_target)
        #     # exit()



# Np, nangp1, lmax, Nd = 420, 9, 4, 12
# shape = (Np, nangp1, Nd, 6)

# blocks  = np.arange(Np * nangp1 * lmax, dtype=np.int32)      # blx
# threads = np.arange(Nd * Nd * 6,        dtype=np.int32)      # thx

# for blx in blocks:                       # ⇢ chooses (n, α, ℓ)
#     g          = blx // lmax             # “strip off” ℓ  →  g∈[0,Np·nangp1)
#     base       = g * Nd * 6              # starting linear position of this (n,α) slice

#     for thx in threads:                  # ⇢ chooses (i, j, μ)
#         μ   = thx % 6                    # 0…5  (fastest–varying)
#         tmp = thx // 6                   # 0…143
#         j   = tmp % Nd                   # 0…11
#         i   = tmp // Nd                  # 0…11

#         # ---- linear indices into dgdteta ---------------------------------
#         index_dgdteta_i = base + i * 6 + μ
#         index_dgdteta_j = base + j * 6 + μ

#         # ---- 4-D multi-indices (Np, nangp1, Nd, 6) -----------------------
#         multi_i = np.unravel_index(index_dgdteta_i, shape)   # (n, α,  i, μ)
#         multi_j = np.unravel_index(index_dgdteta_j, shape)   # (n, α,  j, μ)
#         print(multi_i, multi_j)
#         #  … use multi_i & multi_j (or the linear idxs) as needed …




print('done')
