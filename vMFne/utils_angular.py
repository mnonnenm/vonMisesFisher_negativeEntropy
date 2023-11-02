import numpy as np

def cart2spherical(x):
    if x.shape[0] == 2:
        phi = np.arctan2(x[1],x[0])
        return phi
    elif x.shape[0] == 3:
        xx, yy, zz  = x[0], x[1], x[2]
        r = np.sqrt(xx**2 + yy**2 + zz**2)
        th = np.arccos(zz / r)
        ph = np.arctan2(yy,xx)
        phi = np.stack([th, ph], axis=0)
        return phi

def spherical2cart(phi):
    th, ph = phi[0], phi[1]
    xx = np.sin(th) * np.cos(ph)
    yy = np.sin(th) * np.sin(ph)
    zz = np.cos(th)
    return np.stack([xx,yy,zz], axis=0)

def spherical_rotMat(phi):
    assert len(phi) == 2
    th, ph = phi[0], phi[1]        
    Myy = np.array([[ np.cos(th), 0, np.sin(th)],
                    [          0, 1,          0],
                    [ -np.sin(th),0, np.cos(th)]])
    Mzz = np.array([[np.cos(ph),-np.sin(ph),0],
                    [np.sin(ph), np.cos(ph),0],
                    [         0,          0,1]])
    return Mzz.dot(Myy)