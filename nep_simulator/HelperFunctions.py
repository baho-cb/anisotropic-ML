import numpy as np
import cupy as cp
import torch
from EnergyModel import EnergyModel
import gsd.hoomd
from sklearn import tree
from scipy.spatial import cKDTree as KDTree

def wrap_pbc(x, Box):
    delta = np.where(x > 0.5 * Box, x- Box, x)
    delta = np.where(delta <- 0.5 * Box, Box + delta, delta)
    return delta

def get_dp(q,t,dt):
    """  WORKS (TESTED)
    Happens at first step of NVT-Angular

    - orientations are quats with firs term scalar
    - torks are vectors (relativized in body space)
    - dt is time step

    return q*t which is
    /*! \param a quat
    \param b vector

    Multiplication is quaternion multiplication. The vector is promoted to a quaternion (0,b)

    \returns The quaternion (a.s * b.s − dot(a.v, b.v), a.s*b.v + b.s * a.v + cross(a.v, b.v)).
    scalar part = − dot(a.v, b.v)
    vector part = a.s*b.v + cross(a.v, b.v)
    """
    N_rows = len(q)

    res_s = -np.sum(q[:,1:]*t,axis=1)

    qs = q[:,0].reshape(-1,1)

    cr = np.cross(q[:,1:],t)
    res_v = cr + qs*t

    res = np.zeros((N_rows,4))
    res[:,0] = res_s
    res[:,1:] = res_v
    res = res*dt
    return res

def get_dp_gpu(q,t,dt):
    """  WORKS (TESTED)
    Happens at first step of NVT-Angular

    - orientations are quats with firs term scalar
    - torks are vectors (relativized in body space)
    - dt is time step

    return q*t which is
    /*! \param a quat
    \param b vector

    Multiplication is quaternion multiplication. The vector is promoted to a quaternion (0,b)

    \returns The quaternion (a.s * b.s − dot(a.v, b.v), a.s*b.v + b.s * a.v + cross(a.v, b.v)).
    scalar part = − dot(a.v, b.v)
    vector part = a.s*b.v + cross(a.v, b.v)
    """
    N_rows = len(q)

    res_s = -cp.sum(q[:,1:]*t,axis=1)

    qs = q[:,0].reshape(-1,1)

    cr = cp.cross(q[:,1:],t)
    res_v = cr + qs*t

    res = cp.zeros((N_rows,4))
    res[:,0] = res_s
    res[:,1:] = res_v
    res = res*dt
    return res

def permutation1(p,q,Izz,dt):
    """ WORKS (TESTED)
    This is in the first step of rotational integration
    It is the first permutation if(!z_zero)
    p,q are both quats with first term scalar
    """
    N_rows = len(p)
    p3 = np.zeros((N_rows,4))
    q3 = np.zeros((N_rows,4))

    p3[:,0] = np.copy(-p[:,3])
    p3[:,1] = np.copy(p[:,2])
    p3[:,2] = np.copy(-p[:,1])
    p3[:,3] = np.copy(p[:,0])

    q3[:,0] = np.copy(-q[:,3])
    q3[:,1] = np.copy(q[:,2])
    q3[:,2] = np.copy(-q[:,1])
    q3[:,3] = np.copy(q[:,0])

    phi3 = ((1.0/4.0)/Izz)*np.sum(p*q3,axis=1)
    cphi3 = np.cos(0.5*dt*phi3)
    sphi3 = np.sin(0.5*dt*phi3)

    p = cphi3.reshape(-1,1)*p + sphi3.reshape(-1,1)*p3
    q = cphi3.reshape(-1,1)*q + sphi3.reshape(-1,1)*q3

    return p,q

def permutation1_gpu(p,q,Izz,dt):
    """ WORKS (TESTED)
    This is in the first step of rotational integration
    It is the first permutation if(!z_zero)
    p,q are both quats with first term scalar
    """
    N_rows = len(p)
    p3 = cp.zeros((N_rows,4))
    q3 = cp.zeros((N_rows,4))

    p3[:,0] = cp.copy(-p[:,3])
    p3[:,1] = cp.copy(p[:,2])
    p3[:,2] = cp.copy(-p[:,1])
    p3[:,3] = cp.copy(p[:,0])

    q3[:,0] = cp.copy(-q[:,3])
    q3[:,1] = cp.copy(q[:,2])
    q3[:,2] = cp.copy(-q[:,1])
    q3[:,3] = cp.copy(q[:,0])

    phi3 = ((1.0/4.0)/Izz)*cp.sum(p*q3,axis=1)
    cphi3 = cp.cos(0.5*dt*phi3)
    sphi3 = cp.sin(0.5*dt*phi3)

    p = cphi3.reshape(-1,1)*p + sphi3.reshape(-1,1)*p3
    q = cphi3.reshape(-1,1)*q + sphi3.reshape(-1,1)*q3

    return p,q

def permutation1_gpu2(p,q,Izz,dt):
    """ WORKS (TESTED)
    This is in the first step of rotational integration
    It is the first permutation if(!z_zero)
    p,q are both quats with first term scalar

    self.angmom,self.orientations = permutation1_gpu2(self.angmom,self.orientations,self.moi[2],self.dt)
    """
    N_rows = len(p)
    p3 = cp.zeros((N_rows,4))
    q3 = cp.zeros((N_rows,4))

    p3[:,0] = -p[:,3]
    p3[:,1] = p[:,2]
    p3[:,2] = -p[:,1]
    p3[:,3] = p[:,0]

    q3[:,0] = -q[:,3]
    q3[:,1] = q[:,2]
    q3[:,2] = -q[:,1]
    q3[:,3] = q[:,0]

    phi3 = ((1.0/4.0)/Izz)*cp.sum(p*q3,axis=1) # (Np,)

    cphi3 = cp.cos(0.5*dt*phi3)
    sphi3 = cp.sin(0.5*dt*phi3)

    p = cphi3.reshape(-1,1)*p + sphi3.reshape(-1,1)*p3
    q = cphi3.reshape(-1,1)*q + sphi3.reshape(-1,1)*q3

    return p,q

def permutation2(p,q,Iyy,dt):
    """ WORKS (TESTED)
    This is in the 1st step of rotational integration
    It is the 2nd permutation if(!y_zero)
    p,q are both quats with first term scalar

    p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
    q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
    phi2 = Scalar(1./4.)/I.y*dot(p,q2);
    cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
    sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

    p=cphi2*p+sphi2*p2;
    q=cphi2*q+sphi2*q2;
    """
    N_rows = len(p)
    p2 = np.zeros((N_rows,4))
    q2 = np.zeros((N_rows,4))

    p2[:,0] = np.copy(-p[:,2])
    p2[:,1] = np.copy(-p[:,3])
    p2[:,2] = np.copy(p[:,0])
    p2[:,3] = np.copy(p[:,1])

    q2[:,0] = np.copy(-q[:,2])
    q2[:,1] = np.copy(-q[:,3])
    q2[:,2] = np.copy(q[:,0])
    q2[:,3] = np.copy(q[:,1])


    phi2 = ((1.0/4.0)/Iyy)*np.sum(p*q2,axis=1)
    cphi2 = np.cos(0.5*dt*phi2)
    sphi2 = np.sin(0.5*dt*phi2)

    p = cphi2.reshape(-1,1)*p + sphi2.reshape(-1,1)*p2
    q = cphi2.reshape(-1,1)*q + sphi2.reshape(-1,1)*q2

    return p,q

def permutation2_gpu(p,q,Iyy,dt):
    """ WORKS (TESTED)
    This is in the 1st step of rotational integration
    It is the 2nd permutation if(!y_zero)
    p,q are both quats with first term scalar

    p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
    q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
    phi2 = Scalar(1./4.)/I.y*dot(p,q2);
    cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
    sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

    p=cphi2*p+sphi2*p2;
    q=cphi2*q+sphi2*q2;
    """
    N_rows = len(p)
    p2 = cp.zeros((N_rows,4))
    q2 = cp.zeros((N_rows,4))

    p2[:,0] = cp.copy(-p[:,2])
    p2[:,1] = cp.copy(-p[:,3])
    p2[:,2] = cp.copy(p[:,0])
    p2[:,3] = cp.copy(p[:,1])

    q2[:,0] = cp.copy(-q[:,2])
    q2[:,1] = cp.copy(-q[:,3])
    q2[:,2] = cp.copy(q[:,0])
    q2[:,3] = cp.copy(q[:,1])


    phi2 = ((1.0/4.0)/Iyy)*cp.sum(p*q2,axis=1)
    cphi2 = cp.cos(0.5*dt*phi2)
    sphi2 = cp.sin(0.5*dt*phi2)

    p = cphi2.reshape(-1,1)*p + sphi2.reshape(-1,1)*p2
    q = cphi2.reshape(-1,1)*q + sphi2.reshape(-1,1)*q2

    return p,q

def permutation2_gpu2(p,q,Iyy,dt):
    """ WORKS (TESTED)
    This is in the 1st step of rotational integration
    It is the 2nd permutation if(!y_zero)
    p,q are both quats with first term scalar

    p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
    q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
    phi2 = Scalar(1./4.)/I.y*dot(p,q2);
    cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
    sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

    p=cphi2*p+sphi2*p2;
    q=cphi2*q+sphi2*q2;
    """
    N_rows = len(p)
    p2 = cp.zeros((N_rows,4))
    q2 = cp.zeros((N_rows,4))

    p2[:,0] = -p[:,2]
    p2[:,1] = -p[:,3]
    p2[:,2] = p[:,0]
    p2[:,3] = p[:,1]

    q2[:,0] = -q[:,2]
    q2[:,1] = -q[:,3]
    q2[:,2] = q[:,0]
    q2[:,3] = q[:,1]


    phi2 = ((1.0/4.0)/Iyy)*cp.sum(p*q2,axis=1)
    cphi2 = cp.cos(0.5*dt*phi2)
    sphi2 = cp.sin(0.5*dt*phi2)

    p = cphi2.reshape(-1,1)*p + sphi2.reshape(-1,1)*p2
    q = cphi2.reshape(-1,1)*q + sphi2.reshape(-1,1)*q2

    return p,q

def permutation3(p,q,Ixx,dt):
    """ WORKS (TESTED)
    This is in the 1st step of rotational integration
    It is the 3rd permutation if(!x_zero)
    p,q are both quats with first term scalar

    p1 = quat<Scalar>(-p.v.x,vec3<Scalar>(p.s,p.v.z,-p.v.y));
    q1 = quat<Scalar>(-q.v.x,vec3<Scalar>(q.s,q.v.z,-q.v.y));
    phi1 = Scalar(1./4.)/I.x*dot(p,q1);
    cphi1 = slow::cos(m_deltaT*phi1);
    sphi1 = slow::sin(m_deltaT*phi1);

    p=cphi1*p+sphi1*p1;
    q=cphi1*q+sphi1*q1;
    """
    N_rows = len(p)
    p1 = np.zeros((N_rows,4))
    q1 = np.zeros((N_rows,4))

    p1[:,0] = np.copy(-p[:,1])
    p1[:,1] = np.copy(p[:,0])
    p1[:,2] = np.copy(p[:,3])
    p1[:,3] = np.copy(-p[:,2])

    q1[:,0] = np.copy(-q[:,1])
    q1[:,1] = np.copy(q[:,0])
    q1[:,2] = np.copy(q[:,3])
    q1[:,3] = np.copy(-q[:,2])



    phi1 = ((1.0/4.0)/Ixx)*np.sum(p*q1,axis=1)
    cphi1 = np.cos(dt*phi1)
    sphi1 = np.sin(dt*phi1)

    p = cphi1.reshape(-1,1)*p + sphi1.reshape(-1,1)*p1
    q = cphi1.reshape(-1,1)*q + sphi1.reshape(-1,1)*q1

    return p,q

def permutation3_gpu(p,q,Ixx,dt):
    """ WORKS (TESTED)
    This is in the 1st step of rotational integration
    It is the 3rd permutation if(!x_zero)
    p,q are both quats with first term scalar

    p1 = quat<Scalar>(-p.v.x,vec3<Scalar>(p.s,p.v.z,-p.v.y));
    q1 = quat<Scalar>(-q.v.x,vec3<Scalar>(q.s,q.v.z,-q.v.y));
    phi1 = Scalar(1./4.)/I.x*dot(p,q1);
    cphi1 = slow::cos(m_deltaT*phi1);
    sphi1 = slow::sin(m_deltaT*phi1);

    p=cphi1*p+sphi1*p1;
    q=cphi1*q+sphi1*q1;
    """
    N_rows = len(p)
    p1 = cp.zeros((N_rows,4))
    q1 = cp.zeros((N_rows,4))

    p1[:,0] = cp.copy(-p[:,1])
    p1[:,1] = cp.copy(p[:,0])
    p1[:,2] = cp.copy(p[:,3])
    p1[:,3] = cp.copy(-p[:,2])

    q1[:,0] = cp.copy(-q[:,1])
    q1[:,1] = cp.copy(q[:,0])
    q1[:,2] = cp.copy(q[:,3])
    q1[:,3] = cp.copy(-q[:,2])



    phi1 = ((1.0/4.0)/Ixx)*cp.sum(p*q1,axis=1)
    cphi1 = cp.cos(dt*phi1)
    sphi1 = cp.sin(dt*phi1)

    p = cphi1.reshape(-1,1)*p + sphi1.reshape(-1,1)*p1
    q = cphi1.reshape(-1,1)*q + sphi1.reshape(-1,1)*q1

    return p,q

def permutation3_gpu2(p,q,Ixx,dt):
    """ WORKS (TESTED)
    This is in the 1st step of rotational integration
    It is the 3rd permutation if(!x_zero)
    p,q are both quats with first term scalar

    p1 = quat<Scalar>(-p.v.x,vec3<Scalar>(p.s,p.v.z,-p.v.y));
    q1 = quat<Scalar>(-q.v.x,vec3<Scalar>(q.s,q.v.z,-q.v.y));
    phi1 = Scalar(1./4.)/I.x*dot(p,q1);
    cphi1 = slow::cos(m_deltaT*phi1);
    sphi1 = slow::sin(m_deltaT*phi1);

    p=cphi1*p+sphi1*p1;
    q=cphi1*q+sphi1*q1;
    """
    N_rows = len(p)
    p1 = cp.zeros((N_rows,4))
    q1 = cp.zeros((N_rows,4))

    p1[:,0] = -p[:,1]
    p1[:,1] = p[:,0]
    p1[:,2] = p[:,3]
    p1[:,3] = -p[:,2]

    q1[:,0] = -q[:,1]
    q1[:,1] = q[:,0]
    q1[:,2] = q[:,3]
    q1[:,3] = -q[:,2]



    phi1 = ((1.0/4.0)/Ixx)*cp.sum(p*q1,axis=1)
    cphi1 = cp.cos(dt*phi1)
    sphi1 = cp.sin(dt*phi1)

    p = cphi1.reshape(-1,1)*p + sphi1.reshape(-1,1)*p1
    q = cphi1.reshape(-1,1)*q + sphi1.reshape(-1,1)*q1

    return p,q

def renormalize_quat(q):
    """
    At the end of the first step of the rotation integration
    Good for stability
    q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));

    template < class Real >
    DEVICE inline Real norm2(const quat<Real>& a)
    {
    return (a.s*a.s + dot(a.v,a.v));
    }

    """
    #  q_norm = np.sum(q*q,axis=1) was wrong in the previous version
    q_norm = np.sqrt(np.sum(q*q,axis=1))
    q = q/q_norm.reshape(-1,1)
    return q

def renormalize_quat_gpu(q):
    """
    At the end of the first step of the rotation integration
    Good for stability
    q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));

    template < class Real >
    DEVICE inline Real norm2(const quat<Real>& a)
    {
    return (a.s*a.s + dot(a.v,a.v));
    }

    """
    #  q_norm = np.sum(q*q,axis=1) was wrong in the previous version
    q_norm = cp.sqrt(cp.sum(q*q,axis=1))
    q = q/q_norm.reshape(-1,1)
    return q

def quaternion_multiplication(a,b):
    """
    Is not commutative
    (a.s*b.s - dot(a.v, b.v),  a.s*b.v + b.s * a.v + cross(a.v,b.v));
    Scalar part : a.s*b.s - dot(a.v, b.v)
    Vector part : a.s*b.v + b.s * a.v + cross(a.v,b.v)
    """
    res = np.zeros_like(a)

    s1 = a[:,0]*b[:,0]
    s2 = -np.sum(a[:,1:]*b[:,1:],axis=1)
    scalar = s1+s2

    v1 = a[:,0].reshape(-1,1)*b[:,1:]
    v2 = b[:,0].reshape(-1,1)*a[:,1:]
    v3 = np.cross(a[:,1:],b[:,1:])
    v = v1 + v2 + v3

    res[:,0] = scalar
    res[:,1:] = v

    return res
# get_dp(1.0,2.0,3.0)
def quaternion_multiplication_gpu(a,b):
    """
    Is not commutative
    (a.s*b.s - dot(a.v, b.v),  a.s*b.v + b.s * a.v + cross(a.v,b.v));
    Scalar part : a.s*b.s - dot(a.v, b.v)
    Vector part : a.s*b.v + b.s * a.v + cross(a.v,b.v)
    """
    res = cp.zeros_like(a)

    s1 = a[:,0]*b[:,0]
    s2 = -cp.sum(a[:,1:]*b[:,1:],axis=1)
    scalar = s1+s2

    v1 = a[:,0].reshape(-1,1)*b[:,1:]
    v2 = b[:,0].reshape(-1,1)*a[:,1:]
    v3 = cp.cross(a[:,1:],b[:,1:])
    v = v1 + v2 + v3

    res[:,0] = scalar
    res[:,1:] = v

    return res
# get_dp(1.0,2.0,3.0)

def rotate_torks_to_body_frame(q,t):
    """
    HOOMD gets the torks in box frame and converts (rotate) them into body frame
    with t = rotate(conj(q),t); where t is tork vector and q is orientation quat

    conj : return quat<Real>(a.s, -a.v);
    rotate: (a quat, b vector)
    return (a.s*a.s - dot(a.v, a.v)) *b + 2*a.s*cross(a.v,b) + 2*dot(a.v,b)*a.v;
    """
    q = np.copy(-q)
    q[:,0] = np.copy(-q[:,0])

    coef1 = q[:,0]*q[:,0] - np.sum(q[:,1:]*q[:,1:],axis=1)
    term1 = coef1.reshape(-1,1)*t

    term2 = 2.0*q[:,0].reshape(-1,1)*np.cross(q[:,1:],t)
    term3 = 2.0*np.sum(q[:,1:]*t,axis=1).reshape(-1,1)*q[:,1:]

    res = term1 + term2 + term3

    return res
def rotate_torks_to_body_frame_gpu(q,t):
    """
    HOOMD gets the torks in box frame and converts (rotate) them into body frame
    with t = rotate(conj(q),t); where t is tork vector and q is orientation quat

    conj : return quat<Real>(a.s, -a.v);
    rotate: (a quat, b vector)
    return (a.s*a.s - dot(a.v, a.v)) *b + 2*a.s*cross(a.v,b) + 2*dot(a.v,b)*a.v;
    """
    q = cp.copy(-q)
    q[:,0] = cp.copy(-q[:,0])

    coef1 = q[:,0]*q[:,0] - cp.sum(q[:,1:]*q[:,1:],axis=1)
    term1 = coef1.reshape(-1,1)*t

    term2 = 2.0*q[:,0].reshape(-1,1)*cp.cross(q[:,1:],t)
    term3 = 2.0*cp.sum(q[:,1:]*t,axis=1).reshape(-1,1)*q[:,1:]

    res = term1 + term2 + term3

    return res

def reconstruct_top_pos_from_orientations(q,t):
    """
    After rotating the orientation quaternions using the code copied from hoomd
    I need to place top positions again using updated quaternions and updated COM positions

    Here I'm writing and testing the function that is supposed to do that
    this is the same as rotate(quat,vector) function in python
    """

    coef1 = q[:,0]*q[:,0] - np.sum(q[:,1:]*q[:,1:],axis=1)
    term1 = coef1.reshape(-1,1)*t


    term2 = 2.0*q[:,0].reshape(-1,1)*np.cross(q[:,1:],t)
    term3 = 2.0*np.sum(q[:,1:]*t,axis=1).reshape(-1,1)*q[:,1:]

    res = term1 + term2 + term3

    return res

def rotate(q,v):
    """
    rotate vector v by quat t, same as reconstruct_top_pos_from_orientations
    """
    coef1 = q[:,0]*q[:,0] - np.sum(q[:,1:]*q[:,1:],axis=1)
    term1 = coef1.reshape(-1,1)*v

    term2 = 2.0*q[:,0].reshape(-1,1)*np.cross(q[:,1:],v)
    term3 = 2.0*np.sum(q[:,1:]*v,axis=1).reshape(-1,1)*q[:,1:]

    res = term1 + term2 + term3

    return res
def rotate_gpu(q,v):
    """
    rotate vector v by quat t, same as reconstruct_top_pos_from_orientations
    """
    coef1 = q[:,0]*q[:,0] - cp.sum(q[:,1:]*q[:,1:],axis=1)
    term1 = coef1.reshape(-1,1)*v

    term2 = 2.0*q[:,0].reshape(-1,1)*cp.cross(q[:,1:],v)
    term3 = 2.0*cp.sum(q[:,1:]*v,axis=1).reshape(-1,1)*q[:,1:]

    res = term1 + term2 + term3

    return res

@cp.fuse
def _rotate_gpu_element(qw, qx, qy, qz, x, y, z):
    """
    Elementwise function that rotates a single 3D vector (x,y,z)
    by a single quaternion (qw, qx, qy, qz).
    """
    # Cross product q_xyz x v
    cross_x = qy*z - qz*y
    cross_y = qz*x - qx*z
    cross_z = qx*y - qy*x

    # Dot product q_xyz . v
    dot = qx*x + qy*y + qz*z

    # Scalar factor
    a = qw*qw - (qx*qx + qy*qy + qz*qz)

    # Final rotated vector:
    rx = a*x + 2.0*(qw*cross_x + dot*qx)
    ry = a*y + 2.0*(qw*cross_y + dot*qy)
    rz = a*z + 2.0*(qw*cross_z + dot*qz)
    return rx, ry, rz

def rotate_gpu2(q, v):
    """
    Rotate vectors `v` (N,3) by quaternions `q` (4) using fused elementwise ops.
    """
    # Unpack quaternion (q_w, q_x, q_y, q_z) and vectors (x, y, z)
    # qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    x,  y,  z      = v[:, 0], v[:, 1], v[:, 2]

    # Apply fused rotation
    rx, ry, rz = _rotate_gpu_element(qw, qx, qy, qz, x, y, z)

    # Repack results
    return cp.stack([rx, ry, rz], axis=-1)

def rotate_gpu_qvar(q, v):
    """
    Rotate vectors `v` (N,3) by quaternions `q` (N,4) using fused elementwise ops.
    """
    # Unpack quaternion (q_w, q_x, q_y, q_z) and vectors (x, y, z)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x,  y,  z      = v[:, 0], v[:, 1], v[:, 2]

    # Apply fused rotation
    rx, ry, rz = _rotate_gpu_element(qw, qx, qy, qz, x, y, z)

    # Repack results
    return cp.stack([rx, ry, rz], axis=-1)


def quat_from_two_vectors(v_orig,v):
    """
    My vectors are normalized

    v_original is the vector that happens when orientation quaternion is [1.0,0.0,0.0,0.0]
    https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    Quaternion q;
    rotation from v1 to v2
    vector a = crossproduct(v1, v2);
    q.xyz = a;
    q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);

    Don't forget to normalize it
    """

    vect = np.cross(v_orig,v)
    scalar = 1.0 + np.sum(v_orig*v,axis=1)
    quat = np.zeros((len(vect),4))
    quat[:,0] = scalar
    quat[:,1:] = vect
    quat = renormalize_quat(quat)
    return quat
