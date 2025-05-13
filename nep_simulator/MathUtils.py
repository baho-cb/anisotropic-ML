import cupy as cp

def quaternion_multiplication(a,b):
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