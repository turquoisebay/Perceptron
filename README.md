# Perceptron
Perceptron implementation in Python with a snappy learning visualization

This is a project for practice of Python development. The code implements a classic perceptron. A linear target
function f(x,y), unknown to the perceptron, is synthesized as a line through two random control points. A number of data
points are randomly generated and the function separates the constellation of points, returning -1 if the point is
below the line and 1 if it is above. Training data is generated randomly along with truth values.

The learning algorithm is typical and more information on it can be found by googling. This implementation sets
the weight vector to zeros, which generates a poor hypothesis. At each iteration, a misclassified point is randomly chosen,
and the weight vector is adjusted to correctly classify that point. Eventually the model will converge to correctly classify
all training points. Perceptrons are guaranteed to converge on a solution for any linear classifier.

The learning action is visualized as a matplotlib animation. The true target function is drawn with its control
points, and the training data points are colored according to their set membership - green points are above the target
line and red points are below. The final hypothesis, which correctly separates all the training data, is in light blue,
and the sequence of converging trial hypotheses are in darker blue.

The perceptron can be extended to higher dimensional hyperplanes and to more modern linear classifiers such as
SVMs.
