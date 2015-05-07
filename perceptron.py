__author__ = 'Gene Chalfant'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import animation


class Perceptron:
    """
    Classic perceptron demonstrates the Perceptron Learning Algorithm.

    Given a training dataset, model an unknown function.
    """
    w = np.zeros(3)
    wList = []

    # Evaluate model hypothesis at input points and compare to training set
    def eval(self):
        error = []
        h = self.compute(X)             # compute hypothesis on all input points

        # Build a vector of errors
        if verbose:
            print('y =', y, 'h =', h)
        for yval, hval in zip(y, h):
            if yval == hval:        # Correct classification
                error.append(0)
            elif yval > hval:
                error.append(1)
            elif yval < hval:
                error.append(-1)

        if verbose:
            print("Accuracy on training set =", 100.0*error.count(0)/datasetSize, '%')
        return error

    # Adjust weights
    def updatew(self, mistakes):
        # Find index of a random incorrect (misclassified) value
        bad = np.random.choice([i for i, s in enumerate(mistakes) if s != 0])
        if verbose:
            print('Picking index ', bad, ' to reclassify')

        self.w[0] += mistakes[bad]
        self.w[1] += mistakes[bad] * X.dataset[bad][1]
        self.w[2] += mistakes[bad] * X.dataset[bad][2]
        if verbose:
            print("Weights =", self.w)

    def learn(self):
        nIters = 0
        self.wList = []
        for i in range(iterLimit):
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

    # Compute h(X) = sign(W dot X)
    # Takes an augmented dataset (x[0] = 1 for bias) as input
    def compute(self, inputList):
        h = []
        for point in inputList:
            h.append(int(np.sign(np.dot(self.w, point))))
        return h

    # Compute plot line slope and y-intercept
    def plotParams(self):
        if self.w[2] == 0:          # if denom is 0, nudge it a bit
            self.w[2] = 0.00001
        return -self.w[1]/self.w[2], -self.w[0]/self.w[2]


class Target:
    """
    Function to be modeled, unknown to the perceptron.

    Used for synthesizing training data output values.
    """
    def __init__(self):
        self.p1 = randomPt()
        self.p2 = randomPt()
        self.m = (self.p2[1] - self.p1[1])/(self.p2[0] - self.p1[0])
        self.b = self.p1[1] - self.m * self.p1[0]
        if verbose:
            print("Target function control points = ", self.p1, self.p2)
            print("slope =", self.m, ', y-int =', self.b)

    # Compute target function on list of input data
    def compute(self, inputs):
        results = []
        for (_, x1, x2) in inputs:
            yVal = self.m * x1 + self.b
            if x2 > yVal:
                results.append(1)
            else:
                results.append(-1)
        return results

    def plotParams(self):
        # Reformat points to x-list and y-list for matplotlib
        x = (self.p1[0], self.p2[0])
        y = (self.p1[1], self.p2[1])
        return x, y, self.m, self.b


class Data:
    """
    Input data.
    """
    # Synthesize random 2D + bias input and true output f(x)
    def __init__(self, N):
        self.dataset = []
        for _ in range(N):
            x1, x2 = randomPt()
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
    """
    Collection of static methods for plotting various elements of the visualization.
    """
    @staticmethod
    def plotTarget():
        p1, p2, m, b = f.plotParams()
        ax.scatter(p1, p2, s=40, c='k')               # control points
        ax.plot([-1, 1], [-m+b, m+b], lw=2, c='k')  # line of target function

    @staticmethod
    def plotHypoth():
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


def randomPt():
    """
    Return a random 2D data point.
    """
    return 2*np.random.random()-1, 2*np.random.random()-1

def computeAccuracy(sampSize, target, hypoth):
    """
    Calculate accuracy of the final hypothesis by randomly sampling the input space
    and checking against the target function
    """
    sampList = []
    for _ in range(sampSize):
        x_1, x_2 = randomPt()
        sampList.append((1, x_1, x_2))
    f_x = target.compute(sampList)
    g_x = hypoth.compute(sampList)
    nCorrect = [f-g for f, g in zip(f_x, g_x)].count(0)
    return 100*(nCorrect/sampSize)

# Animation functions to pass to matplotlib
def vizInit():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(hyp[i])
    return line,

#####################################################################
# Project parameters
nRuns = 20            # 1000 number of independent experiments
datasetSize = 100       # 100 size of training dataset
iterLimit = 5000        # Give up on convergence after this many iterations
verbose = False         # Show diagnostics
viz = True              # Show visualization
if nRuns > 20:          # Don't do too many plots, it's tedious
    viz = False

avgIterations = 0.0
avgAccuracy = 0.0           # Probability that

for run in range(nRuns):
    print(run+1, end=') ')

    f = Target()            # Begin with an unknown function
    X = Data(datasetSize)   # Synthesize input dataset
    y = f.compute(X)     # Compute y=f(X) training set
    p = Perceptron()        # Instantiate the model

    nIterations, hyp = p.learn()      # Train the model

    # After training, animate the progression of hypotheses
    if viz:
        style.use('ggplot')
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
        line, = ax.plot([], [], lw=2, c='blue')
        plt.title('Perceptron Learning Algorithm')
        Display.plotTarget()
        Display.plotTrainingData(X, y)
        Display.plotHypoth()       # Plot the final hypothesis
        fps = 45
        frameInt = 1000/fps
        anim = animation.FuncAnimation(
            fig, animate, init_func=vizInit, frames=nIterations-1,
            interval=frameInt, blit=True, repeat=False)

    print(nIterations, 'iterations to converge')
    avgIterations += nIterations
    accuracyThisRun = computeAccuracy(20000, f, p)
    print('Run accuracy =', format(accuracyThisRun, '6.3f'))
    avgAccuracy += accuracyThisRun
    plt.show()

# Average statistics across all runs
avgIterations /= nRuns
print('Average number of iterations to converge =', avgIterations)
avgAccuracy /= nRuns
print('Average hypothesis accuracy =', avgAccuracy)