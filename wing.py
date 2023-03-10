import numpy as np
import matplotlib.pyplot as plt


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
        if self.M > 2:
            for i in np.arange(0, self.M-1):
                xpos = (self.Dchord/4)*(self.M-2-i) / \
                    (self.M-2) + (self.Dchord)*(i/(self.M-2))
                xs[i, :] = self.xoff + xpos
                ys[i, :] = self.Dspan
                zs[i, :] = self.groundDist + self.Ddihedr + \
                    self.airfoil.camber_line(xpos)  # camber_line
        else:
            xs[0, :] = self.Dchord/4 + self.xoff
            ys[0, :] = self.Dspan
            zs[0, :] = self.groundDist + self.Ddihedr + \
                self.airfoil.camber_line(self.Dchord/4)  # camber_line

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

    def solveWingHS(self, alpha0, Umag, dens, solveFun):
        if self.M == 3:
            RHS = np.zeros((self.N-1)*(self.M-2))
            a = np.zeros(((self.N-1)*(self.M-2), (self.N-1)*(self.M-2)))
            b = np.zeros(((self.N-1)*(self.M-2), (self.N-1)*(self.M-2)))
            infMat = np.zeros(((self.N-1)*(self.M-2), (self.N-1)*(self.M-2)))
        elif self.M == 2:
            RHS = np.zeros((self.N-1)*(self.M-1))
            a = np.zeros(((self.N-1)*(self.M-1), (self.N-1)*(self.M-1)))
            b = np.zeros(((self.N-1)*(self.M-1), (self.N-1)*(self.M-1)))
            infMat = np.zeros(((self.N-1)*(self.M-1), (self.N-1)*(self.M-1)))
        else:
            print(f"M is {self.M} did you want to do panels?")
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
        return L, D, Gammas, w_ind

    def solveWingPanels(self, Q, solveFun):
        a_np = np.zeros(((self.N-1)*(self.M-1), (self.N-1)*(self.M-1)))
        b_np = np.zeros(((self.N-1)*(self.M-2), (self.N-1)*(self.M-2)))
        RHS_np = np.zeros((self.N-1)*(self.M-1))

        for i in np.arange(0, (self.N-1)*(self.M-1)):
            lp, kp = divmod(i, (self.M-1))
            if kp == self.M-2:
                RHS_np[i] = 0
                a_np[i, i] = 1
                a_np[i, i-1] = -1
                continue
            RHS_np[i] = - np.dot(Q, self.control_nj[lp, kp])
            for j in np.arange(0, (self.N-1)*(self.M-1)):
                l, k = divmod(j, (self.M-1))
                if k == self.M-2:
                    U, Ustar = solveFun(self.controlP[lp, kp, 0],
                                        self.controlP[lp, kp, 1],
                                        self.controlP[lp, kp, 2],
                                        l, k, self.grid)
                else:
                    U, Ustar = solveFun(self.controlP[lp, kp, 0],
                                        self.controlP[lp, kp, 1],
                                        self.controlP[lp, kp, 2],
                                        l, k, self.grid)

                    l1, k1 = divmod(i, (self.M-2))
                    l2, k2 = divmod(j, (self.M-2))
                    b_np[l1*(self.M-2) - lp + k1, l2 * (self.M-2) - l + k2] =\
                        np.dot(Ustar, self.control_nj[lp, kp])

                a_np[i, j] = np.dot(U, self.control_nj[lp, kp])
        return a_np, b_np, RHS_np

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
