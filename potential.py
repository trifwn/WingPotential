import numpy as np


def vortexL(xp, yp, zp, x1, y1, z1, x2, y2, z2, gamma):
    """Computes the velocities induced at a point xp,yp,zp 
    by a vortex line given its two end points and circulation

    Args:
        xp: x coordinate of point 
        yp: y coordinate of point 
        zp: z coordinate of point 
        x1: x coordinate of line startpoint 
        y1: y coordinate of line startpoint 
        z1: z coordinate of line startpoint 
        x2: x coordinate of line endpoint
        y2: y coordinate of line endpoint
        z2: z coordinate of line endpoint
        gamma: vorticity

    Returns:
        u,v,w: induced velocities
    """
    crossx = (yp - y1) * (zp - z2) - (zp - z1) * (yp-y2)
    crossy = -(xp - x1) * (zp-z2) + (zp - z1) * (xp-x2)
    crossz = (xp-x1) * (yp - y2) - (yp - y1) * (xp-x2)

    cross_mag = crossx**2 + crossy**2 + crossz**2
    r1 = np.sqrt((xp-x1)**2 + (yp-y1)**2 + (zp-z1)**2)
    r2 = np.sqrt((xp-x2)**2 + (yp-y2)**2 + (zp-z2)**2)

    e = 1e-9
    if (r1 < e or r2 < e or cross_mag < e):
        return 0, 0, 0
    r0dr1 = (x2-x1)*(xp - x1) + (y2 - y1)*(yp-y1) + (z2-z1)*(zp-z1)
    r0dr2 = (x2-x1)*(xp - x2) + (y2 - y1)*(yp-y2) + (z2-z1)*(zp-z2)

    K = (gamma / (4 * np.pi * cross_mag)) * (r0dr1/r1 - r0dr2/r2)
    u = K * crossx
    v = K * crossy
    w = K * crossz

    return u, v, w


def voring(x, y, z, i, j, gd, gamma=1):
    """Vorticity Ring Element. Computes the velocities induced at a point x,y,z 
    by a vortex ring given its grid lower corner coordinates 

    Args:
        x: x coordinate of point 
        y: y coordinate of point 
        z: z coordinate of point 
        i: specifies i index of grid (gd[i,j])
        j: specifies j index of grid (gd[i,j])
        gd: grid of geometry
        gamma: Circulation. Defaults to 1 (When we use nondimensional solve).

    Returns:
        U: induced velocities vector
    """
    u1, v1, w1 = vortexL(x, y, z,
                         gd[0, i+1, j], gd[1, i+1, j], gd[2, i+1, j],
                         gd[0, i, j], gd[1, i, j], gd[2, i, j],
                         gamma)
    u2, v2, w2 = vortexL(x, y, z,
                         gd[0, i, j], gd[1, i, j], gd[2, i, j],
                         gd[0, i, j+1], gd[1, i, j+1], gd[2, i, j+1],
                         gamma)
    u3, v3, w3 = vortexL(x, y, z,
                         gd[0, i, j+1], gd[1, i, j+1], gd[2, i, j+1],
                         gd[0, i+1, j+1], gd[1, i + 1, j+1], gd[2, i+1, j+1],
                         gamma)
    u4, v4, w4 = vortexL(x, y, z,
                         gd[0, i+1, j+1], gd[1, i + 1, j+1], gd[2, i+1, j+1],
                         gd[0, i+1, j], gd[1, i+1, j], gd[2, i+1, j],
                         gamma)

    u = u1 + u2 + u3 + u4
    v = v1 + v2 + v3 + v4
    w = w1 + w2 + w3 + w4

    ustar = u2 + u4
    vstar = v2 + v4
    wstar = w2 + w4

    U = np.array((u, v, w))
    Ustar = np.array((ustar, vstar, wstar))
    return U  # , Ustar


def hshoe(x, y, z, i, j, gd, gamma=1):
    """Vorticity Horseshow Element. Computes the velocities induced at a point x,y,z 
    by a horseshow Vortex given its grid lower corner coordinates 

    Args:
        x: x coordinate of point 
        y: y coordinate of point 
        z: z coordinate of point 
        i: specifies i index of grid (gd[i,j])
        j: specifies j index of grid (gd[i,j])
        gd: grid of geometry
        gamma: Circulation. Defaults to 1 (When we use nondimensional solve).

    Returns:
        U: induced velocities vector
    """
    u1, v1, w1 = vortexL(x, y, z,
                         gd[0, i+1, j], gd[1, i + 1, j], gd[2, i+1, j],
                         gd[0, i, j], gd[1, i, j], gd[2, i, j],
                         gamma)
    u2, v2, w2 = vortexL(x, y, z,
                         gd[0, i, j], gd[1, i, j], gd[2, i, j],
                         gd[0, i, j+1], gd[1, i, j+1], gd[2, i, j+1],
                         gamma)
    u3, v3, w3 = vortexL(x, y, z,
                         gd[0, i, j+1], gd[1, i, j+1], gd[2, i, j+1],
                         gd[0, i+1, j+1], gd[1, i + 1, j+1], gd[2, i+1, j+1],
                         gamma)

    u = u1 + u2 + u3
    v = v1 + v2 + v3
    w = w1 + w2 + w3

    ustar = u1 + u3
    vstar = v1 + v3
    wstar = w1 + w3

    U = np.array((u, v, w))
    Ustar = np.array((ustar, vstar, wstar))
    return U, Ustar


def hshoe2(x, y, z, j, k, gd, gamma=1):
    """Vorticity Horseshow Element. Computes the velocities induced at a point x,y,z
    by a horseshow Vortex given its grid lower corner coordinates

    Args:
        x: x coordinate of point
        y: y coordinate of point
        z: z coordinate of point
        i: specifies i index of grid (gd[i,j])
        j: specifies j index of grid (gd[i,j])
        gd: grid of geometry
        gamma: Circulation. Defaults to 1 (When we use nondimensional solve).

    Returns:
        U: induced velocities vector
    """
    u1, v1, w1 = vortexL(x, y, z,
                         gd[j, k+1, 0], gd[j, k+1, 1], gd[j, k+1, 2],
                         gd[j, k, 0], gd[j, k, 1], gd[j, k, 2],
                         gamma)
    u2, v2, w2 = vortexL(x, y, z,
                         gd[j, k, 0], gd[j, k, 1], gd[j, k, 2],
                         gd[j+1, k, 0], gd[j+1, k, 1], gd[j+1, k, 2],
                         gamma)
    u3, v3, w3 = vortexL(x, y, z,
                         gd[j+1, k, 0], gd[j+1, k, 1], gd[j+1, k, 2],
                         gd[j+1, k+1, 0], gd[j + 1, k+1, 1], gd[j+1, k+1, 2],
                         gamma)

    u = u1 + u2 + u3
    v = v1 + v2 + v3
    w = w1 + w2 + w3

    U = np.array((u, v, w))
    return U


def hshoeSL(x, y, z, i, j, gd, gamma=1):
    """Slanted Horseshoe Element

    Args:
        x: _description_
        y: _description_
        z: _description_
        i: _description_
        j: _description_
        gd: _description_
        gamma (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    u1, v1, w1 = vortexL(x, y, z,
                         gd[0, i+2, j], gd[1, i+2, j], gd[2, i+2, j],
                         gd[0, i+1, j], gd[1, i+1, j], gd[2, i+1, j],
                         gamma)
    u2, v2, w2 = vortexL(x, y, z,
                         gd[0, i+1, j], gd[1, i+1, j], gd[2, i+1, j],
                         gd[0, i, j], gd[1, i, j], gd[2, i, j],
                         gamma)
    u3, v3, w3 = vortexL(x, y, z,
                         gd[0, i, j], gd[1, i, j], gd[2, i, j],
                         gd[0, i, j+1], gd[1, i, j+1], gd[2, i, j+1],
                         gamma)
    u4, v4, w4 = vortexL(x, y, z,
                         gd[0, i, j+1], gd[1, i, j+1], gd[2, i, j+1],
                         gd[0, i+1, j+1], gd[1, i + 1, j+1], gd[2, i+1, j+1],
                         gamma)
    u5, v5, w5 = vortexL(x, y, z,
                         gd[0, i+1, j+1], gd[1, i + 1, j+1], gd[2, i+1, j+1],
                         gd[0, i+2, j+1], gd[1, i + 2, j+1], gd[2, i+2, j+1],
                         gamma)

    u = u1 + u2 + u3 + u4 + u5
    v = v1 + v2 + v3 + v4 + v5
    w = w1 + w2 + w3 + w4 + w5

    ustar = u1 + u2 + u4 + u5
    vstar = v1 + v2 + v4 + v5
    wstar = w1 + w2 + w4 + w5

    U = np.array((u, v, w))
    Ustar = np.array((ustar, vstar, wstar))
    return U, Ustar


def hshoeSL2(x, y, z, i, j, gd, gamma=1):
    """Slanted Horseshoe Element To Work with Panels

    Args:
        x : _description_
        y : _description_
        z : _description_
        i : _description_
        j : _description_
        gd: _description_
        gamma (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    u1, v1, w1 = vortexL(x, y, z,
                         gd[j, i+2, 0], gd[j, i+2, 1], gd[j, i+2, 2],
                         gd[j, i+1, 0], gd[j, i+1, 1], gd[j, i+1, 2],
                         gamma)
    u2, v2, w2 = vortexL(x, y, z,
                         gd[j, i+1, 0], gd[j, i+1, 1], gd[j, i+1, 2],
                         gd[j, i, 0], gd[j, i, 1], gd[j, i, 2],
                         gamma)
    u3, v3, w3 = vortexL(x, y, z,
                         gd[j, i, 0], gd[j, i, 1], gd[j, i, 2],
                         gd[j+1, i, 0], gd[j+1, i, 1], gd[j+1, i, 2],
                         gamma)
    u4, v4, w4 = vortexL(x, y, z,
                         gd[j+1, i, 0], gd[j+1, i, 1], gd[j+1, i, 2],
                         gd[j+1, i+1, 0], gd[j+1, i+1, 1], gd[j+1, i+1, 2],
                         gamma)
    u5, v5, w5 = vortexL(x, y, z,
                         gd[j+1, i+1, 0], gd[j+1, i+1, 1], gd[j+1, i+1, 2],
                         gd[j+1, i+2, 0], gd[j+1, i+2, 1], gd[j+1, i+2, 2],
                         gamma)

    u = u1 + u2 + u3 + u4 + u5
    v = v1 + v2 + v3 + v4 + v5
    w = w1 + w2 + w3 + w4 + w5

    U = np.array((u, v, w))
    return U
