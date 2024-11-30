import numpy as np
from scipy.spatial.transform import Rotation as RR

def random_quat():
    rands = np.random.uniform(size=3)
    quat = np.array([np.sqrt(1.0-rands[0])*np.sin(2*np.pi*rands[1]),
            np.sqrt(1.0-rands[0])*np.cos(2*np.pi*rands[1]),
            np.sqrt(rands[0])*np.sin(2*np.pi*rands[2]),
            np.sqrt(rands[0])*np.cos(2*np.pi*rands[2])])
    return quat


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

def quat_from_two_vectors(v_orig,v):
    """
    Input vectors are normalized

    v_original is the vector that happens when orientation quaternion is [1.0,0.0,0.0,0.0]
    https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    Quaternion q;
    rotation from v1 to v2
    vector a = crossproduct(v1, v2);
    q.xyz = a;
    q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
    """

    vect = np.cross(v_orig,v)
    scalar = 1.0 + np.sum(v_orig*v,axis=1)
    quat = np.zeros((len(vect),4))
    quat[:,0] = scalar
    quat[:,1:] = vect
    # print(quat)
    quat = renormalize_quat(quat)
    return quat



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
    q_norm = np.sqrt(np.sum(q*q,axis=1))
    q = q/q_norm.reshape(-1,1)
    return q


def quat_from_axis_angle(axis,angle):
    """
    q.w == cos(angle / 2)
    q.x == sin(angle / 2) * axis.x
    q.y == sin(angle / 2) * axis.y
    q.z == sin(angle / 2) * axis.z

    """
    q = np.zeros((len(angle),4))
    q[:,0] = np.cos(angle*0.5)
    q[:,1] = np.sin(angle*0.5)*axis[:,0]
    q[:,2] = np.sin(angle*0.5)*axis[:,1]
    q[:,3] = np.sin(angle*0.5)*axis[:,2]
    return q


def rotation_matrix_to_quaternion(R):
    """
    Converts a 3x3 rotation matrix to a quaternion (scalar first).

    Parameters:
    R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
    numpy.ndarray: A quaternion [w, x, y, z] corresponding to the rotation matrix.
    """
    # Ensure the matrix is 3x3
    if R.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3.")

    # Compute the trace of the matrix
    t = np.trace(R)

    if t > 0:
        S = np.sqrt(t + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    # Normalize the quaternion
    quaternion = np.array([qw, qx, qy, qz])
    quaternion /= np.linalg.norm(quaternion)

    return quaternion


def spherical_to_cartesian(sph):
    """takes spherical return cartesian trasnlation"""
    N_pair = len(sph)
    trans = np.zeros((N_pair,3))
    sine_polar = np.sin(sph[:,2])

    trans[:,0] = sph[:,0]*sine_polar*np.cos(sph[:,1])
    trans[:,1] = sph[:,0]*sine_polar*np.sin(sph[:,1])
    trans[:,2] = sph[:,0]*np.cos(sph[:,2])

    return trans


def euler_to_rotmax(eulers):
    """takes spherical return cartesian trasnlation"""
    N = len(eulers)
    rot = RR.from_euler('ZXZ',eulers)
    rotmaxs = rot.as_matrix()
    pos = np.zeros((1,3))
    pos[:,0] = 1.0
    # print(pos)
    # exit()
    pos = np.matmul(rotmaxs,np.transpose(pos))
    pos = np.transpose(pos,(0,2,1))
    print(pos.shape)

    return pos






























def dummy_ff():
    pass
