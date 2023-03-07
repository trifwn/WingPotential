import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Wing:
    def __init__(self, airfoil, isSymmetric, span, alpha, beta, Ni, Nj, sweepAngle, dihAngle, chordFun, spanFun):
        self.Dspan = spanFun(span, Ni)
        self.Dchord = chordFun(Ni)
        self.Lamda = sweepAngle * np.pi/180
        self.Gamma = dihAngle*np.pi/180
        if isSymmetric == True:
            self.Dsweep = np.hstack((
                np.linspace((span/2)*np.sin(self.Lamda), 0, int(Ni/2)),
                np.linspace(0, (span/2)*np.sin(self.Lamda), int(Ni/2)),
            ))
            self.Ddihedr = np.hstack((
                np.linspace((span/2)*np.sin(self.Gamma), 0, int(Ni/2)),
                np.linspace(0, (span/2)*np.sin(self.Gamma), int(Ni/2)),
            ))
        else:
            print("Not implemented Yet")
        self.Dwake = 10 * self.Dchord
        self.N = Ni
        self.M = Nj
        self.alpha = alpha * np.pi/180
        self.beta = beta * np.pi/180
        self.airfoil = airfoil
        self.createGrid(plotting=True)

    def set_alpha(self, a):
        self.alpha = a * np.pi/180
        self.createGrid()

    def set_beta(self, b):
        self.beta = b * np.pi/180
        self.createGrid()

    def set_airfoil(self, airfoil):
        self.airfoil = airfoil
        self.createGrid()

    def plotgrid(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in np.arange(0, self.N-1):
            ax.plot(self.airfoil._x_upper * self.Dchord[i] + self.Dsweep[i], np.repeat(
                self.grid[i, 0, 1], len(self.airfoil._y_upper)), self.airfoil._y_upper + self.Ddihedr[i], '-', color='red')
            ax.plot(self.airfoil._x_lower * self.Dchord[i] + self.Dsweep[i], np.repeat(
                self.grid[i, 0, 1], len(self.airfoil._y_upper)), self.airfoil._y_lower + self.Ddihedr[i], '-', color='red')

            for j in np.arange(0, self.M-1):
                p1, p3, p4, p2 = self.panels[i, j, :, :]
                xs = np.reshape([p1[0], p2[0], p3[0], p4[0]], (2, 2))
                ys = np.reshape([p1[1], p2[1], p3[1], p4[1]], (2, 2))
                zs = np.reshape([p1[2], p2[2], p3[2], p4[2]], (2, 2))

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
            xpos = (self.Dchord/4)*(self.M-1-i) / \
                (self.M-1) + (self.Dchord)*(i/(self.M-1))
            xs[i, :] = self.Dsweep + xpos
            ys[i, :] = self.Dspan
            zs[i, :] = self.Ddihedr + self.airfoil.camber_line(xpos)

        xs[-1, :] = self.Dwake
        ys[-1, :] = self.Dspan
        zs[-1, :] = zs[0, :] + self.Dwake * np.sin(self.alpha)

        self.grid = np.array((xs, ys, zs)).T

        self.panels = np.empty((self.N-1, self.M-1, 4, 3))

        for i in np.arange(0, self.N-1):
            for j in np.arange(0, self.M-1):
                self.panels[i, j, 0, :] = self.grid[i+1, j]
                self.panels[i, j, 1, :] = self.grid[i, j]
                self.panels[i, j, 2, :] = self.grid[i, j+1]
                self.panels[i, j, 3, :] = self.grid[i+1, j+1]

        self.nj = np.repeat([(np.sin(self.alpha) * np.cos(self.beta),
                              np.cos(self.alpha) * np.sin(self.beta),
                              np.cos(self.alpha) * np.cos(self.beta))], (self.N-1)*(self.M-1), axis=0)
        if plotting == True:
            self.plotgrid()
