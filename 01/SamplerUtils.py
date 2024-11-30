import numpy as np
import gsd.hoomd
from scipy.spatial import distance
from pyDOE import lhs
from GeometryUtils import *
import cupy as cp
from scipy.spatial.transform import Rotation as R

"""
Always use scalar first quaternions
scipy returns quats with scalar last format we want scalar first
"""

def db_pos():
    x = np.array([
    [1.0,0.0,0.0],
    [-1.0,0.0,0.0],
    [0.0,1.0,0.0],
    [0.0,-1.0,0.0],
    [0.0,0.0,1.0],
    [0.0,0.0,-1.0],
    ])
    return x

def get_LHC_simple(seed,N):
    np.random.seed(seed)
    lhs_normal = lhs(6, samples=N)

    ###-+-+-+-+-+ unreduced +-+-+-+-+-###
    mins = np.array([-3.142,  0.0,  -3.142,  4.3,    -3.142,  0.0])
    maxs = np.array([3.142, 3.142, 3.142, 7.2,    3.142, 3.142])

    ### reduced ###
    # mins = np.array([1.0,0.0,0.0,0.0,-3.142,0.0])
    # maxs = np.array([7.194,5.0,4.0,2.1,+3.142,1.572])

    lhs_ = lhs_normal*(maxs-mins) + mins
    return lhs_

def get_LHC_adjusted(seed,n):
    """
    Assume e1, e2, e3, r, azimuthal, polar, (euler with ZXZ convention as always)
    Convert simple LHC to uniformly distributed euler angles and spherical coordinates
    """
    np.random.seed(seed)
    lhs6 = lhs(6, samples=n)

    lhs6[:,3:] = (lhs6[:,3:]-0.5) * (7.2*2.0)

    dist = np.linalg.norm(lhs6[:,3:],axis=1)
    lhs6 = lhs6[dist<7.2]
    dist = dist[dist<7.2]
    lhs6 = lhs6[dist>4.3]
    dist = dist[dist>4.3]


    quat = np.zeros((len(lhs6),4))
    quat[:,0] = np.sqrt(1.0-lhs6[:,0])*np.sin(2*np.pi*lhs6[:,1])
    quat[:,1] = np.sqrt(1.0-lhs6[:,0])*np.cos(2*np.pi*lhs6[:,1])
    quat[:,2] = np.sqrt(lhs6[:,0])*np.sin(2*np.pi*lhs6[:,2])
    quat[:,3] = np.sqrt(lhs6[:,0])*np.cos(2*np.pi*lhs6[:,2])
    rot = R.from_quat(quat)
    eulers = rot.as_euler('ZXZ')
    lhs6[:,:3] = eulers

    polar = np.arccos(lhs6[:,5] / dist)
    azimuthal = np.arctan2(lhs6[:,4], lhs6[:,3])

    lhs6[:,3] = dist
    lhs6[:,4] = azimuthal
    lhs6[:,5] = polar

    return lhs6




def unred_to_raw(unred):
    """
    Convert unreduced
    e1, e2, e3, r, azimuthal, polar, (euler with ZXZ convention as always)
    to reduced.

    place the two shapes with p1 at origin, q1 elementary quat
    Use the 3 points of the 2nd cube to track it's final orientation
    """
    N_pair = len(unred)
    random_rots = R.random(num=N_pair)
    random_rots = random_rots.as_matrix()

    pts2 = np.array([
    [1.0,0.0,0.0],
    [0.0,1.0,0.0],
    [0.0,0.0,1.0],
    [0.0,0.0,0.0],
    ])
    pts_orig = np.copy(pts2)
    pts_orig = pts_orig[:3]


    ### Create the array of translations
    trans = np.zeros((N_pair,3))
    sine_polar = np.sin(unred[:,5])

    trans[:,0] = unred[:,3]*sine_polar*np.cos(unred[:,4])
    trans[:,1] = unred[:,3]*sine_polar*np.sin(unred[:,4])
    trans[:,2] = unred[:,3]*np.cos(unred[:,5])



    ### Create rotation matrices
    rot_max = np.zeros((N_pair,3,3))
    s1 = np.sin(unred[:,0])
    c1 = np.cos(unred[:,0])
    s2 = np.sin(unred[:,1])
    c2 = np.cos(unred[:,1])
    s3 = np.sin(unred[:,2])
    c3 = np.cos(unred[:,2])

    rot_max[:,0, 0] = c1*c3 - c2*s1*s3  # r11
    rot_max[:,0, 1] = -c1*s3 - c2*c3*s1  # r12
    rot_max[:,0, 2] = s1*s2  # r13
    rot_max[:,1, 0] = c3*s1 + c1*c2*s3  # r21
    rot_max[:,1, 1] = -s1*s3 + c1*c2*c3  # r22
    rot_max[:,1, 2] = -c1*s2  # r23
    rot_max[:,2, 0] = s2*s3  # r31
    rot_max[:,2, 1] = c3*s2  # r32
    rot_max[:,2, 2] = c2  # r33


    pts2 = np.matmul(rot_max,np.transpose(pts2))
    pts2 = np.transpose(pts2,(0,2,1))


    pts2 = pts2 + trans[:, np.newaxis,:]

    ### this is tested ###
    random_rots_mult = random_rots.transpose(0, 2, 1)
    pts2 = np.matmul(pts2, random_rots_mult)
    ### this is tested ###
    trans = pts2[:,-1,:]

    ### find the rotated orientation of the second shape ##
    ### what we actually do here is solving R@X=Y while knowing X (pts before)
    ### and Y points after, -> R = Y@X_inverse and X_inverse is identity
    ### so rotated points already give the rotation matrix

    pts2 = pts2[:,:,:] - pts2[:, -1, :][:, np.newaxis, :]
    ori2_rm = pts2[:,:3,:].transpose(0, 2, 1)


    ## convert rotation matrices to quats for the 2 shapes
    ori2_q = R.from_matrix(ori2_rm).as_quat()
    ori2_q_xyz = np.copy(ori2_q[:,:3])
    ori2_q_w = np.copy(ori2_q[:,3])
    ori2_q[:,0] = ori2_q_w
    ori2_q[:,1:] = ori2_q_xyz

    ori1_q = R.from_matrix(random_rots).as_quat()
    ori1_q_xyz = np.copy(ori1_q[:,:3])
    ori1_q_w = np.copy(ori1_q[:,3])
    ori1_q[:,0] = ori1_q_w
    ori1_q[:,1:] = ori1_q_xyz

    # ori2_q = np.zeros((N_pair,4))
    # ori1_q = np.zeros((N_pair,4))
    # for i in range(N_pair):
    #     ori2_q[i] = rotation_matrix_to_quaternion(ori2_rm[i])
    #     ori1_q[i] = rotation_matrix_to_quaternion(random_rots[i])

    ## p1, q1, p2, q2
    all_raw = np.zeros((N_pair,14))
    all_raw[:,:3] = (np.random.rand(N_pair,3) - 0.5)*5.0
    all_raw[:,3:7] = ori1_q
    # all_raw[:,3] = 1.0
    all_raw[:,7:10] = all_raw[:,:3] + trans
    all_raw[:,10:] = ori2_q

    return all_raw

def raw_to_red(raw):
    """
    reduces the output of unred_to_raw(), just like original code
    """

    self_N_pair = len(raw)

    COM1 = raw[:,:3]
    COM2 = raw[:,7:10]
    QUAT1 = raw[:,3:7]
    QUAT2 = raw[:,10:]

    """
    RELATIVIZE V4
    """

    """ FIND INTERACTING FACE 1 """

    true_faces1 = np.array([
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [-1.0,0.0,0.0],
    [0.0,1.0,0.0],
    [0.0,-1.0,0.0],
    [0.0,0.0,1.0],
    [0.0,0.0,-1.0]
    ])


    true_faces1 = np.tile(true_faces1,(self_N_pair,1))
    q1_faces = np.repeat(QUAT1,7,axis=0)
    com1_faces = np.repeat(COM1,7,axis=0)
    com2_faces = np.repeat(COM2,7,axis=0)
    faces1 = rotate(q1_faces,true_faces1)
    faces1_abs = faces1 + com1_faces
    com2_faces1_rel = com2_faces-faces1_abs
    dist2faces1 = np.linalg.norm(com2_faces1_rel,axis=1)
    dist2faces1 = dist2faces1.reshape(-1,7)
    face1_index = np.argmin(dist2faces1,axis=1)

    """ FIND INTERACTING FACE 2 """

    true_faces2 = np.array([
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [-1.0,0.0,0.0],
    [0.0,1.0,0.0],
    [0.0,-1.0,0.0],
    [0.0,0.0,1.0],
    [0.0,0.0,-1.0]
    ])

    true_faces2 = np.tile(true_faces2,(self_N_pair,1))
    q2_faces = np.repeat(QUAT2,7,axis=0)
    com1_faces = np.repeat(COM1,7,axis=0)
    com2_faces = np.repeat(COM2,7,axis=0)
    faces2 = rotate(q2_faces,true_faces2)
    faces2_abs = faces2 + com2_faces

    com1_faces2_rel = faces2_abs - com1_faces
    dist2faces2 = np.linalg.norm(com1_faces2_rel,axis=1)


    # if(np.max(dist2faces2)>7.5):
    #     print(np.max(dist2faces2))
    #     exit()
    # print(np.max(dist2faces2))

    dist2faces2 = dist2faces2.reshape(-1,7)
    face2_index = np.argmin(dist2faces2,axis=1)

    """ Rotate everything such that interacting face1 is [1.0,0.0,0.0] """
    right_true = np.array([1.0,0.0,0.0])
    right_true = np.tile(right_true,(self_N_pair,1))

    faces1i = faces1.reshape(-1,7,3)
    faces1_inter = np.zeros((self_N_pair,3))
    for i in range(self_N_pair):
        faces1_inter[i] = np.copy(faces1i[i,face1_index[i],:])

    # faces1_inter_test = np.copy(faces1i[np.arange(self_N_pair), face1_index, :])
    # faces1_inter_cpp = np.loadtxt('../139/comp.txt')

    q_rot1u = quat_from_two_vectors(faces1_inter,right_true)
    q_rot1 = np.repeat(q_rot1u,7,axis=0)
    faces1_r1 = rotate(q_rot1,faces1)
    faces2_r1 = rotate(q_rot1,com1_faces2_rel)

    face1p_index = (face1_index + 2)%7
    face1p_index[face1p_index==0] = 1

    forward_true = np.array([0.0,1.0,0.0])
    forward_true = np.tile(forward_true,(self_N_pair,1))
    faces1p_inter = np.zeros((self_N_pair,3))
    faces1ri = faces1_r1.reshape(-1,7,3)

    for i in range(self_N_pair):
        faces1p_inter[i] = np.copy(faces1ri[i,face1p_index[i],:])

    q_rot2u = quat_from_two_vectors(faces1p_inter,forward_true)
    q_rot2 = np.repeat(q_rot2u,7,axis=0)
    faces1_r2 = rotate(q_rot2,faces1_r1)
    faces2_r2 = rotate(q_rot2,faces2_r1)

    faces2ri = faces2_r2.reshape(-1,7,3)
    faces2_r2_com = faces2ri[:,0,:]

    multiplier = np.ones_like(faces2_r2)
    y_signs = np.sign(faces2_r2_com[:,1])
    z_signs = np.sign(faces2_r2_com[:,2])

    # multiplier_force = np.ones((self_N_pair,3))
    # multiplier_force[:,1] = y_signs
    # multiplier_force[:,2] = z_signs
    #
    # multiplier_tork = np.ones((self_N_pair,3))
    # multiplier_tork[:,0] = y_signs*z_signs
    # multiplier_tork[:,1] = z_signs
    # multiplier_tork[:,2] = y_signs

    y_signs = np.repeat(y_signs,7)
    z_signs = np.repeat(z_signs,7)

    multiplier[:,1] = y_signs
    multiplier[:,2] = z_signs

    faces2_r2 = faces2_r2*multiplier

    faces2ri = faces2_r2.reshape(-1,7,3)


    faces2_r2_com = faces2ri[:,0,:]


    switch_index = np.where(faces2_r2_com[:,2]>faces2_r2_com[:,1])
    switch_index = switch_index[0]
    will_switch = np.zeros(self_N_pair,dtype=int)
    will_switch[switch_index] = 1

    # will_switch = np.ma.make_mask(will_switch, shrink=False)
    # faces2ri[will_switch, :, [1, 2]] = faces2ri[will_switch, :, [2, 1]]
    # print(faces2ri[will_switch, :, :][:,:,[1,2]].shape)

    for i in range(self_N_pair):
        if(i in switch_index):
            faces2ri[i,:,[1,2]] = faces2ri[i,:,[2,1]]
            # force_r2[i,[1,2]] = force_r2[i,[2,1]]
            # tork_r2[i,[1,2]] = -tork_r2[i,[2,1]]
            # tork_r2[i,0] = -tork_r2[i,0]

    faces2_r2 = faces2ri.reshape(-1,3)
    faces2_inter = np.zeros((self_N_pair,3))
    for i in range(self_N_pair):
        faces2_inter[i] = np.copy(faces2ri[i,face2_index[i],:])

    faces2_r2_com = faces2ri[:,0,:]

    faces2_r2_com = faces2ri[:,0,:]
    faces2_r2_com7 = np.repeat(faces2_r2_com,7,axis=0)

    faces2_r2_relcom2 = faces2_r2 - faces2_r2_com7


    faces2_inter_relcom2 = np.zeros((self_N_pair,3))
    faces2_r2_relcom2_3d = faces2_r2_relcom2.reshape(-1,7,3)

    for i in range(self_N_pair):
        faces2_inter_relcom2[i] = np.copy(faces2_r2_relcom2_3d[i,face2_index[i],:])

    """ Be careful we take the cosine from the opposite direction """

    """ Very rarely arccos won't work because the x dimension will be larger than 1 due to numerical errors"""
    faces2_inter_relcom2[faces2_inter_relcom2[:,0]<-1.0,0] = -1.0
    faces2_inter_relcom2[faces2_inter_relcom2[:,0]>1.0,0] = 1.0

    xcos_angle2 = np.arccos(-faces2_inter_relcom2[:,0]) # 0 ile pi arasi gelir bu
    yztan_angle2 = np.arctan2(-faces2_inter_relcom2[:,2],-faces2_inter_relcom2[:,1])


    """
    - Only the last angle of 2 is left
    (1) Find the quat that will rotate face2_inter_relcom2 towards [-1,0,0]
    (2) Apply the quat to the faces2_r2_relcom2
    (3) See that opposite of the face2_inter_relcom2 is towards [1,0,0]
    (4) Find the angle to rotate other faces towards z and y axis
    """

    true_left = np.array([-1.0,0.0,0.0])
    true_left = np.tile(true_left,(len(faces2_inter_relcom2),1))
    q21 = quat_from_two_vectors(faces2_inter_relcom2,true_left)
    q21 = np.repeat(q21,7,axis=0)
    faces2_r2_r21_relcom2 = rotate(q21,faces2_r2_relcom2)

    faces2p_inter = np.zeros((self_N_pair,3))
    faces2r21_3d = faces2_r2_r21_relcom2.reshape(-1,7,3)

    # if(self.timestep==1):
    #     xcp = np.loadtxt('../139/dat.txt')
    #     acos = np.loadtxt('../139/acos.txt')
    #     print(xcos_angle2-acos)
    #     # print(xcp[:20])
    #     # print(faces2_r2_r21_relcom2[:20])
    #     exit()

    for i in range(self_N_pair):
        ff = np.copy(faces2r21_3d[i])
        ff = ff[1:]
        ff = ff[np.abs(ff[:,0])<0.001]
        ff = ff[ff[:,2]>=0.0]
        ff = ff[ff[:,1]>0.000]
        if(len(ff)!=1):
            print("ppl")
            exit()
            print(i)
            print(pair0[i],pair1[i])
            print(faces2r21_3d[i])
            # print(ff)
            #exit()
            self.dumpConfigDebug(min_d+1.0)
            xcos_angle2[i] = 1.6
            faces2p_inter[i] = np.array([1.0,1.0,1.0])
        else:
            #print('no ppl')
            faces2p_inter[i] = ff[0]
    ff2 = faces2r21_3d[:, 1:, :]

    # Filter elements based on conditions using boolean indexing
    condition1 = np.abs(ff2[:,:,0]) < 0.001
    condition2 = ff2[:,:,2] >= 0.0
    condition3 = ff2[:,:,1] > 0.0
    # Apply conditions using logical AND
    condition_all = condition1 & condition2 & condition3
    # print(condition1.shape)
    # Apply the combined condition to filter ff
    ff_filtered = ff2[condition_all]
    # print(ff_filtered.shape,faces2p_inter.shape,yztan_angle2.shape)


    # print(ff_filtered)
    ## last angle is correct
    last_angle = np.arctan2(faces2p_inter[:,2],faces2p_inter[:,1])
    reduced_configs = np.zeros((len(last_angle),6))
    reduced_configs[:,:3] = faces2_r2_com


    reduced_configs[:,3] = xcos_angle2
    reduced_configs[:,4] = yztan_angle2
    reduced_configs[:,5] = last_angle

    return reduced_configs

def unred_to_12pts(unred):
    table_chunk = unred
    factor = 1.7
    self_pos = db_pos()
    self_pos = self_pos*factor
    self_Np = len(self_pos)

    self_Npair = len(table_chunk)
    Npair = len(table_chunk)
    N_pair = len(table_chunk)

    index_pos1 = np.concatenate([np.arange(i, i+6) for i in range(0, Npair*12, 12)])
    index_pos2 = np.concatenate([np.arange(i+6, i+12) for i in range(0, Npair*12, 12)])

    pos2 = np.copy(self_pos)
    pos1 = np.copy(self_pos)
    pos2 = pos2.astype(np.float32)
    pos1 = pos1.astype(np.float32)


    ### Create the array of translations
    trans = np.zeros((N_pair,3))
    sine_polar = np.sin(table_chunk[:,5])

    trans[:,0] = table_chunk[:,3]*sine_polar*np.cos(table_chunk[:,4])
    trans[:,1] = table_chunk[:,3]*sine_polar*np.sin(table_chunk[:,4])
    trans[:,2] = table_chunk[:,3]*np.cos(table_chunk[:,5])

    ### Create rotation matrices
    rot_max = np.zeros((N_pair,3,3))
    s1 = np.sin(table_chunk[:,0])
    c1 = np.cos(table_chunk[:,0])
    s2 = np.sin(table_chunk[:,1])
    c2 = np.cos(table_chunk[:,1])
    s3 = np.sin(table_chunk[:,2])
    c3 = np.cos(table_chunk[:,2])

    rot_max[:,0, 0] = c1*c3 - c2*s1*s3  # r11
    rot_max[:,0, 1] = -c1*s3 - c2*c3*s1  # r12
    rot_max[:,0, 2] = s1*s2  # r13
    rot_max[:,1, 0] = c3*s1 + c1*c2*s3  # r21
    rot_max[:,1, 1] = -s1*s3 + c1*c2*c3  # r22
    rot_max[:,1, 2] = -c1*s2  # r23
    rot_max[:,2, 0] = s2*s3  # r31
    rot_max[:,2, 1] = c3*s2  # r32
    rot_max[:,2, 2] = c2  # r33


    ## rotate and translate
    pos2 = np.matmul(rot_max,np.transpose(pos2))
    pos2 = np.transpose(pos2,(0,2,1))

    pos2 = pos2 + trans[:, np.newaxis,:]
    pos1 = np.tile(pos1,(Npair,1,1))


    pos12 = np.zeros((N_pair*12,3))
    pos1 = np.reshape(pos1,(N_pair*6,3))
    pos2 = np.reshape(pos2,(N_pair*6,3))

    pos12[index_pos1] = pos1
    pos12[index_pos2] = pos2

    q_new = R.random(num=N_pair)
    q_new = q_new.as_quat()

    q_xyz = np.copy(q_new[:,:3])
    q_w = np.copy(q_new[:,3])
    q_new[:,0] = q_w
    q_new[:,1:] = q_xyz
    q_new = np.repeat(q_new,self_Np*2,axis=0)

    pos12 = rotate(q_new,pos12)
    pos12 = np.reshape(pos12,(N_pair,12,3))

    return pos12

def pts12_to_cld(pts12):
    Np = len(pts12)
    p0 = pts12[:,:6,:]
    p1 = pts12[:,6:,:]
    p0 = np.tile(p0,(1,6,1))
    p1 = np.repeat(p1,6,axis=1)
    cld = np.linalg.norm(p1-p0,axis=2)
    cld = np.sort(cld,axis=-1)
    return cld

def calc_en_raw(raw):
    shapename = 'cube_structure.gsd'
    with gsd.hoomd.open(name=shapename, mode='r') as f:
        pos0 = f[0].particles.position

    # pos0 = db_pos()
    Np = len(pos0)
    quat1 = np.tile(raw[3:7],(Np,1))
    quat2 = np.tile(raw[10:],(Np,1))

    p1 = rotate(quat1,pos0)
    p2 = rotate(quat2,pos0)
    p1 += raw[:3]
    p2 += raw[7:10]

    dev1 = cp.cuda.Device(1)
    dev1.use()
    pos2 = cp.asarray(p2)
    pos1 = cp.asarray(p1)
    pos2 = cp.tile(pos2,(len(pos2),1))
    pos1 = cp.repeat(pos1,len(pos1),axis=0)
    delta = pos1 - pos2
    delta = cp.linalg.norm(delta,axis=1)


    b = np.power(2.0,1./6.)
    eps = 5.0
    p = 2.0
    sigma = 1.1
    pot = (b-delta)/sigma
    pot *= pot*eps
    pot[delta>b] = 0.0

    energy_per_pair = cp.sum(pot).get()
    return energy_per_pair

def calc_en_unred(unred):
    table_chunk = unred
    shapename = 'cube_structure.gsd'
    with gsd.hoomd.open(name=shapename, mode='r') as f:
        self_pos = f[0].particles.position
    # self_pos = db_pos()
    self_Np = len(self_pos)
    pos0_gpu = np.copy(self_pos)
    self_Npair = len(table_chunk)
    Npair = len(table_chunk)

    pos0_gpu = pos0_gpu.astype(np.float32)
    pos0_gpu = np.tile(pos0_gpu,(Npair,1,1))
    self_pos0_gpu = np.tile(pos0_gpu,(1,self_Np,1))
    gpu_id = 1
    if(gpu_id>-0.5):
        self_device = cp.cuda.Device(gpu_id)
        self_device.use()
        self_pos0_gpu = cp.asarray(self_pos0_gpu)

    N_pair = len(table_chunk)
    pos2 = np.copy(self_pos)
    pos2 = pos2.astype(np.float32)


    ### Create the array of translations
    trans = np.zeros((N_pair,3))
    sine_polar = np.sin(table_chunk[:,5])

    trans[:,0] = table_chunk[:,3]*sine_polar*np.cos(table_chunk[:,4])
    trans[:,1] = table_chunk[:,3]*sine_polar*np.sin(table_chunk[:,4])
    trans[:,2] = table_chunk[:,3]*np.cos(table_chunk[:,5])

    ### Create rotation matrices
    rot_max = np.zeros((N_pair,3,3))
    s1 = np.sin(table_chunk[:,0])
    c1 = np.cos(table_chunk[:,0])
    s2 = np.sin(table_chunk[:,1])
    c2 = np.cos(table_chunk[:,1])
    s3 = np.sin(table_chunk[:,2])
    c3 = np.cos(table_chunk[:,2])

    rot_max[:,0, 0] = c1*c3 - c2*s1*s3  # r11
    rot_max[:,0, 1] = -c1*s3 - c2*c3*s1  # r12
    rot_max[:,0, 2] = s1*s2  # r13
    rot_max[:,1, 0] = c3*s1 + c1*c2*s3  # r21
    rot_max[:,1, 1] = -s1*s3 + c1*c2*c3  # r22
    rot_max[:,1, 2] = -c1*s2  # r23
    rot_max[:,2, 0] = s2*s3  # r31
    rot_max[:,2, 1] = c3*s2  # r32
    rot_max[:,2, 2] = c2  # r33

    rot_max = cp.asarray(rot_max)
    pos2 = cp.asarray(pos2)
    trans = cp.asarray(trans)

    ## rotate and translate
    pos2 = cp.matmul(rot_max,cp.transpose(pos2))
    pos2 = cp.transpose(pos2,(0,2,1))


    pos2 = pos2 + trans[:, cp.newaxis,:]
    pos2 = cp.repeat(pos2,self_Np,axis=1)

    delta = cp.linalg.norm(pos2-self_pos0_gpu,axis=2)


    b = np.power(2.0,1./6.)
    eps = 5.0
    p = 2.0
    sigma = 1.1
    pot = (b-delta)/sigma
    pot *= pot*eps
    pot[delta>b] = 0.0

    energy_per_pair = cp.sum(pot,axis=1).get()
    return energy_per_pair


def calc_en_red(red):
    shapename = 'cube_structure.gsd'
    with gsd.hoomd.open(name=shapename, mode='r') as f:
        pos = f[0].particles.position
    # pos = db_pos()
    pos1 = np.copy(pos)
    pos2 = np.copy(pos)
    pos1 = pos1.astype(np.float32)
    pos2 = pos2.astype(np.float32)
    N_pair = len(red)

    pos2 = np.copy(pos1)
    N_bead = len(pos1)

    pos1 = np.tile(pos1,(N_pair,1))
    pos2 = np.tile(pos2,(N_pair,1))

    q = quat_from_axis_angle(np.tile(np.array([1.0,0.0,0.0]),(N_pair,1)),red[:,5])
    q = np.repeat(q,N_bead,axis=0)

    pos2 = rotate(q,pos2)
    wx = np.cos(red[:,3])

    #### 1 ####
    wy = np.sqrt((1-wx**2)/(np.tan(red[:,4])**2 + 1))
    revert_index = np.where(np.abs(red[:,4])>(np.pi*0.5))[0]
    wy[revert_index] *= -1.0
    wz = wy*np.tan(red[:,4])

    #### 1 ####
    w = np.column_stack((wx,wy,wz))

    q = quat_from_two_vectors(np.tile(np.array([1.0,0.0,0.0]),(N_pair,1)),w)
    q = np.repeat(q,N_bead,axis=0)
    pos2 = rotate(q,pos2)
    trans = np.repeat(red[:,:3],N_bead,axis=0)
    pos2 = pos2 + trans
    pos1 = np.reshape(pos1,(N_pair,N_bead,3))
    pos2 = np.reshape(pos2,(N_pair,N_bead,3))

    dev1 = cp.cuda.Device(1)
    dev1.use()
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    pos1 = cp.asarray(pos1)
    pos2 = cp.asarray(pos2)


    pos1 = cp.tile(pos1,(1,N_bead,1))
    pos2 = cp.repeat(pos2,N_bead,axis=1)
    delta = pos1 - pos2
    delta = cp.linalg.norm(delta,axis=2)

    b = np.power(2.0,1./6.)
    eps = 5.0
    p = 2.0
    sigma = 1.1
    pot = (b-delta)/sigma
    pot *= pot*eps
    pot[delta>b] = 0.0

    energy_per_pair = cp.sum(pot,axis=1).get()
    return energy_per_pair










































def dummy_func():
    pass
