import numpy as np
from scipy.spatial.transform import Rotation as R
import cupy as cp


def pts_tetrahedron():
    """
    th_v1's points are perfect.
    It is centered at origin with moi diagonalized its vertex indices
    can be used to calculate mid points of its faces

    can place the points to face centers of vertices both are 4
    we did faces with the cube but vertices can be more descriptive as they are further away from each other
    [[-0.45000038 -0.45000002 -0.45000002]
    [-0.45000038  0.45000002  0.45000002]
    [ 0.44999966 -0.45000002  0.45000002]
    [ 0.44999966  0.45000002 -0.45000002]]
    """
    x = np.array([
    [-0.45,-0.45,-0.45],
    [-0.45,0.45,0.45],
    [0.45,-0.45,0.45],
    [0.45,0.45,-0.45],
    ])
    return x


def pts_tetrahedron_vertices():
    x = np.array([
        [-1.35, -1.35, 1.35],
        [-1.35, 1.35, -1.35],
        [1.35, -1.35, -1.35],
        [1.35, 1.35, 1.35]
    ])
    return x 

def pts_tetrahedron_edges():
    x = np.array([
        [-1.35,0.0,0.0],
        [0.0,-1.35,0.0],
        [0.0,0.0,1.35],
        [0.0,0.0,-1.35],
        [0.0,1.35,0.0],
        [1.35,0.0,0.0]
    ])
    return x 

def rotate(q,v):
    """
    rotate vector v by quat t, same as reconstruct_top_pos_from_orientations
    (N,4),(N,3) -> input shapes
    """
    coef1 = q[:,0]*q[:,0] - np.sum(q[:,1:]*q[:,1:],axis=1)
    term1 = coef1.reshape(-1,1)*v

    term2 = 2.0*q[:,0].reshape(-1,1)*np.cross(q[:,1:],v)
    term3 = 2.0*np.sum(q[:,1:]*v,axis=1).reshape(-1,1)*q[:,1:]

    res = term1 + term2 + term3

    return res




def unred_to_8pts(unred,factor,pts_type):
    table_chunk = unred
    if(pts_type=='vertices'):
        self_pos = pts_tetrahedron_vertices()
        n_pts = 4 
    elif(pts_type=='faces'):
        self_pos = pts_tetrahedron()
        n_pts = 4 
    elif(pts_type=='edges'):
        self_pos = pts_tetrahedron_edges()
        n_pts =6     
    else:
        print('invalid pts_type')
        exit()

    n2pts = 2*n_pts    
    self_pos = self_pos*factor
    self_Np = len(self_pos)

    self_Npair = len(table_chunk)
    Npair = len(table_chunk)
    N_pair = len(table_chunk)

    index_pos1 = np.concatenate([np.arange(i, i+n_pts) for i in range(0, Npair*n2pts, n2pts)])
    index_pos2 = np.concatenate([np.arange(i+n_pts, i+n2pts) for i in range(0, Npair*n2pts, n2pts)])

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


    pos12 = np.zeros((N_pair*n2pts,3))
    pos1 = np.reshape(pos1,(N_pair*n_pts,3))
    pos2 = np.reshape(pos2,(N_pair*n_pts,3))

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
    pos12 = np.reshape(pos12,(N_pair,n2pts,3))

    return pos12
