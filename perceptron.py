__author__ = 'Gene Chalfant'

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import animation


class Perceptron:
    iterLimit = 5000     # Give up on convergence after this many iterations
    w = np.zeros(3)
    wList = []

    # Evaluate model hypothesis at input points and compare to training set
    def eval(self):
        error = []
        h = self.compute()     # compute h(x)

        # Build a vector of errors
        for yval, hval in zip(y, h):
            if verbose:
                print('y =', yval, 'h =', hval)
            if yval == hval:        # Correct classification
                error.append(0)
            elif yval > hval:
                error.append(1)
            elif yval < hval:
                error.append(-1)
        if verbose:
            print("Eval =", 100.0*error.count(0)/datasetSize, '%')
        return error

    # Adjust weights
    def updatew(self, e):
        # Find index of a random incorrect (misclassified) value
        m = np.random.choice([i for i, x in enumerate(e) if x != 0])

        self.w[0] += e[m]
        self.w[1] += e[m] * X.dataset[m][1]
        self.w[2] += e[m] * X.dataset[m][2]

    def learn(self):
        nIters = 0
        self.wList = []
        for i in range(self.iterLimit):
            nIters += 1
            err = self.eval()
            if verbose:
                print('Errors (-1 if h < y, 1 if h > y):', err)
            if err.count(0) == len(err):
                break
            self.updatew(err)

            # Record weights for visualization
            m, b = self.plotParams()
            self.wList.append(((-1, 1), (-m+b, m+b)))
        return nIters, self.wList

    # Compute h(X) = sign(W.X)
    def compute(self):
        h = []
        for x in X:
            h.append(int(np.sign(np.dot(self.w, x))))
        return h

    # Compute plot line slope and y-intercept
    def plotParams(self):
        if self.w[2] == 0:          # if denom is 0, nudge it a bit
            self.w[2] = 0.00001
        return -self.w[1]/self.w[2], -self.w[0]/self.w[2]


class Target:
    def __init__(self):
        self.point1 = randomXY()
        self.point2 = randomXY()
        self.m = (self.point2[1] - self.point1[1])/(self.point2[0] - self.point1[0])
        self.b = self.point1[1] - self.m * self.point1[0]
        if verbose:
            print("Target function control points = ", self.point1, self.point2)
            print("Slope =", self.m, ', y-int =', self.b)

    # Compute target function on one input
    def compute(self, inputs):
        f_x = []
        for (_, x1, x2) in inputs:
            yVal = self.m * x1 + self.b
            if x2 > yVal:
                f_x.append(1)
            else:
                f_x.append(-1)
        return f_x

    def plotParams(self):
        # control points to generate target function
        x = [self.point1[0], self.point2[0]]
        y = [self.point1[1], self.point2[1]]

        # slope and y-intercept to generate plot line intercepts at margins
        m = self.m
        b = self.b
        return x, y, m, b


class Data:
    # Synthesize random 2D + bias input and true output f(x)
    def __init__(self, N):
        self.dataset = []
        for _ in range(N):
            x1, x2 = randomXY()
            self.dataset.append((1, x1, x2))
        if verbose:
            print("Input data: ", self.dataset)

    # def list(self):
    def __iter__(self):
        return iter(self.dataset)

    def __str__(self):
        return str(self.dataset)

    # Reformat data for plotting
    def plotParams(self):
        return zip(*self.dataset)


class Display:
    @staticmethod
    def plotTarget():
        x, y, m, b = f.plotParams()
        ax.scatter(x, y, s=40, c='k')
        ax.plot([-1, 1], [-m+b, m+b], lw=2, c='k')

    @staticmethod
    def plotHypoth(ax):
        m, b = p.plotParams()
        ax.plot([-1, 1], [-m+b, m+b], lw=2, c='cyan')

    @staticmethod
    def plotTrainingData(X, yVals):
        _, X1, X2 = X.plotParams()
        x1_above = []
        x2_above = []
        x1_below = []
        x2_below = []
        for i in range(len(yVals)):
            if yVals[i] == 1:
                x1_above.append(X1[i])
                x2_above.append(X2[i])
            if yVals[i] == -1:
                x1_below.append(X1[i])
                x2_below.append(X2[i])
        ax.scatter(x1_above, x2_above, s=30, c='g')
        ax.scatter(x1_below, x2_below, s=30, c='r')


# Return a random point in 2D space
def randomXY():
    return 2*np.random.random()-1, 2*np.random.random()-1

# Project parameters
nRuns = 10           # 1000 number of independent experiments
datasetSize = 50    # 100 size of training dataset
verbose = False     # Show diagnostics
viz = True          # Show visualization
if nRuns > 20:      # Just do stats
    viz = False


def vizInit():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(hyp[i])
    return line,

avgIterations = 0.0
avgProbError = 0.0

for run in range(nRuns):
    print(run+1, end=') ')

    f = Target()            # Begin with an unknown function
    X = Data(datasetSize)   # Synthesize input dataset
    y = f.compute(X)        # Compute y=f(X) training set
    p = Perceptron()        # Instantiate the model

    n, hyp = p.learn()
    if viz:
        # Set up visualization
        style.use('ggplot')
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
        line, = ax.plot([], [], lw=2, c='blue')
        plt.title('Perceptron Learning Algorithm')
        Display.plotTarget()
        Display.plotTrainingData(X, y)
        Display.plotHypoth(ax)       # Plot the final hypothesis
        fps = 45
        frameInt = 1000/fps
        anim = animation.FuncAnimation(
            fig, animate, init_func=vizInit, frames=n-1,
            interval=frameInt, blit=True, repeat=False)
    avgIterations += n
    print(n, 'iterations')
    plt.show()

# Compute stats across entire experiment
avgIterations /= nRuns
print('Average number of iterations to converge =', avgIterations)

# Then use the Perceptron Learning Algorithm to find the model hypothesis
# e = Experiment()
# for run in range(nRuns):
#     if verbosity > 0:
#         print("Test Run ", run)
#     e.run()
#     plt.show()

# average_iterations += perceptron.run()
# average_probability_of_error += perceptron.probability_of_error

# average_iterations /= RUNS
# average_probability_of_error = 1.0 - (average_probability_of_error / RUNS)
#
# print("On average it takes ", average_iterations, " to converge.\n")
# print("Average probability of error: ", average_probability_of_error)