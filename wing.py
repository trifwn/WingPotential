import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Wing:
    def __init__(self, airfoil, isSymmetric, span, alpha, beta, Ni, Nj, sweepOffset, dihAngle, chordFun, spanFun, ground_clearence=5):
        self.N = Ni
        self.M = Nj
        self.Dspan = spanFun(span, Ni)
        self.Dchord = chordFun(Ni)
        self.Gamma = dihAngle * np.pi/180
        self.isSymmetric = isSymmetric
        self.span = span
        self.groundDist = ground_clearence
        if isSymmetric == True:
            self.xoff = np.abs(self.Dspan - span/2) * (sweepOffset / (span/2))
            self.Ddihedr = np.abs(self.Dspan - span/2)*np.sin(self.Gamma)
        else:
            self.xoff = self.Dspan * sweepOffset / span
            self.Ddihedr = self.Dspan*np.sin(self.Gamma)

        self.Dwake = 20 * self.Dchord
        self.alpha = alpha * np.pi/180
        self.beta = beta * np.pi/180
        self.airfoil = airfoil
        self.createGrid(plotting=False)

    def set_alpha(self, a):
        self.alpha = a * np.pi/180
        self.createGrid()

    def set_beta(self, b):
        self.beta = b * np.pi/180
        self.createGrid()

    def set_groundClearence(self, d):
        self.groundDist = d
        self.createGrid()

    def set_airfoil(self, airfoil):
        self.airfoil = airfoil
        self.createGrid()

    def set_offset(self, o):
        if self.isSymmetric == True:
            self.xoff = np.abs(self.Dspan - self.span/2) * (o / (self.span/2))
        else:
            self.xoff = self.Dspan * o / self.span
        self.createGrid()

    def plotgrid(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in np.arange(0, self.N-1):
            ax.plot(self.airfoil._x_upper * self.Dchord[i] + self.xoff[i], np.repeat(
                self.grid[i, 0, 1], len(self.airfoil._y_upper)), self.groundDist + self.airfoil._y_upper + self.Ddihedr[i], '-', color='red')
            ax.plot(self.airfoil._x_lower * self.Dchord[i] + self.xoff[i], np.repeat(
                self.grid[i, 0, 1], len(self.airfoil._y_upper)), self.groundDist + self.airfoil._y_lower + self.Ddihedr[i], '-', color='red')

            for j in np.arange(0, self.M-1):
                p1, p3, p4, p2 = self.panels[i, j, :, :]
                xs = np.reshape([p1[0], p2[0], p3[0], p4[0]], (2, 2))
                ys = np.reshape([p1[1], p2[1], p3[1], p4[1]], (2, 2))
                zs = np.reshape([p1[2], p2[2], p3[2], p4[2]], (2, 2))
                ax.scatter(*self.controlP[i, j, :], color='k')
                ax.plot_wireframe(xs, ys, zs, linewidth=0.5)
        ax.set_title('Grid')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.axis('scaled')
        ax.view_init(30, 150)

    def createGrid(self, plotting=False):
        xs = np.empty((self.M, self.N))
        ys = np.empty((self.M, self.N))
        zs = np.empty((self.M, self.N))

        for i in np.arange(0, self.M-1):
            xpos = (self.Dchord/4)*(self.M-2-i) / \
                (self.M-2) + (self.Dchord)*(i/(self.M-2))
            xs[i, :] = self.xoff + xpos
            ys[i, :] = self.Dspan
            zs[i, :] = self.groundDist + self.Ddihedr + \
                self.airfoil.camber_line(xpos)  # camber_line

        xs[-1, :] = self.Dwake
        ys[-1, :] = self.Dspan
        zs[-1, :] = zs[-2, :] + (self.Dwake-xs[-2, :]) * np.tan(self.alpha)

        self.grid = np.array((xs, ys, zs)).T

        self.panels = np.empty((self.N-1, self.M-1, 4, 3))
        self.controlP = np.empty((self.N-1, self.M-1, 3))
        self.control_nj = np.empty((self.N-1, self.M-1, 3))
        self.wingArea = 0
        for i in np.arange(0, self.N-1):
            self.wingArea += (self.grid[i+1, 0, 1] -
                              self.grid[i, 0, 1]) * self.Dchord[i]
            for j in np.arange(0, self.M-1):
                self.panels[i, j, 0, :] = self.grid[i+1, j]
                self.panels[i, j, 1, :] = self.grid[i, j]
                self.panels[i, j, 2, :] = self.grid[i, j+1]
                self.panels[i, j, 3, :] = self.grid[i+1, j+1]

                self.controlP[i, j, 0] = (self.grid[i, j][0] + self.grid[i+1, j][0])/2 \
                    + 3/4 * ((self.grid[i, j+1][0] + self.grid[i+1, j+1][0])/2 -
                             (self.grid[i, j][0] + self.grid[i+1, j][0])/2)
                self.controlP[i, j, 1] = (self.grid[i, j][1] + self.grid[i+1, j][1])/2 \
                    + 1/2 * ((self.grid[i, j+1][1] + self.grid[i+1, j+1][1])/2 -
                             (self.grid[i, j][1] + self.grid[i+1, j][1])/2)
                self.controlP[i, j, 2] = (self.grid[i, j][2] + self.grid[i+1, j][2])/2 \
                    + 1/2 * ((self.grid[i, j+1][2] + self.grid[i+1, j+1][2])/2 -
                             (self.grid[i, j][2] + self.grid[i+1, j][2])/2)
                Ak = self.panels[i, j, 0, :] - self.panels[i, j, 2, :]
                Bk = self.panels[i, j, 1, :] - self.panels[i, j, 3, :]
                cross = np.cross(Ak, Bk)
                self.control_nj[i, j, :] = cross / np.linalg.norm(cross)
        self.nj = np.repeat([(np.sin(self.alpha) * np.cos(self.beta),
                              np.cos(self.alpha) * np.sin(self.beta),
                              np.cos(self.alpha) * np.cos(self.beta))], (self.N-1)*(self.M-1), axis=0)
        if plotting == True:
            self.plotgrid()

    def solveWing(self, alpha0, Umag, dens, solveFun):
        if self.M > 2:
            RHS = np.zeros((self.N-1)*(self.M-2))
            a = np.zeros(((self.N-1)*(self.M-2), (self.N-1)*(self.M-2)))
            b = np.zeros(((self.N-1)*(self.M-2), (self.N-1)*(self.M-2)))
            infMat = np.zeros(((self.N-1)*(self.M-2), (self.N-1)*(self.M-2)))
        else:
            RHS = np.zeros((self.N-1)*(self.M-1))
            a = np.zeros(((self.N-1)*(self.M-1), (self.N-1)*(self.M-1)))
            b = np.zeros(((self.N-1)*(self.M-1), (self.N-1)*(self.M-1)))
            infMat = np.zeros(((self.N-1)*(self.M-1), (self.N-1)*(self.M-1)))
        w_ind = np.zeros((self.N-1, 3))
        L_pan = np.zeros((self.N-1))
        D_pan = np.zeros((self.N-1))

        for i in np.arange(0, self.N-1):
            RHS[i] = np.pi * (self.alpha - alpha0) * self.Dchord[i] * Umag
            for j in np.arange(0, self.N-1):
                k = 0
                U, Ustar = solveFun((self.grid[i, 0, 0] + self.grid[i+1, 0, 0])/2,
                                    (self.grid[i, 0, 1] +
                                     self.grid[i+1, 0, 1])/2,
                                    (self.grid[i, 0, 2] +
                                     self.grid[i+1, 0, 2])/2,
                                    k, j, self.grid)
                a[i, j] = np.dot(self.nj[i], U)
                b[i, j] = np.dot(self.nj[i], Ustar)
                infMat[i, j] = - a[i, j] * self.Dchord[i] * np.pi
            infMat[i, i] = infMat[i, i] + 1

        Gammas = np.linalg.solve(infMat, RHS)
        w_ind = np.matmul(b, Gammas)
        # w_ind2 = np.matmul(a, Gammas)

        for i in np.arange(0, self.N-1):
            L_pan[i] = dens * Umag * Gammas[i] * \
                (self.grid[i+1, 0, 1] - self.grid[i, 0, 1])
            D_pan[i] = - dens * w_ind[i] * Gammas[i] * \
                (self.grid[i+1, 0, 1] - self.grid[i, 0, 1])

        L = np.sum(L_pan)
        D = np.sum(D_pan)
        return L, D, Gammas, w_ind, L_pan

    def InducedVelocities(self, fun, i, j, gammas):
        Us = 0
        Uss = 0
        for l in np.arange(0, self.N-1):
            for k in np.arange(0, self.M-1):
                U, Ustar = fun(self.controlP[i, j, 0],
                               self.controlP[i, j, 1],
                               self.controlP[i, j, 2],
                               l, k, self.grid,
                               gamma=gammas[l, k])
                Us = Us + U
                Uss = Uss + Ustar
        return Us, Uss
