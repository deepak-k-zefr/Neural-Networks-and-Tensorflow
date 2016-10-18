# Neural Networks Demystified
# Part 2: Forward Propagation
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch
from scipy.optimize import minimize
from scipy import optimize
from scipy.optimize import minimize
## ----------------------- Part 1 ---------------------------- ##
import numpy as np

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X / np.amax(X, axis=0)
y = y / 100  # Max test score is 100


## ----------------------- Part 2 ---------------------------- ##

class Neural_Network(object):
    def __init__(self):
        # Define Hyper parameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def costFunction(self,X,y):
        self.yHat=self.forward(X)
        J=0.5* sum((y-self.yHat)**2)
        return J

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


NN=Neural_Network()
import numpy as np

X=np.array(([3,5],[5,1],[10,2]),dtype=float)
y=np.array(([75],[82],[93]),dtype=float)
# Normalize
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

#print NN.forward(X)
#print NN.costFunction(X,y)
djdW1,djdW2= NN.costFunctionPrime(X,y)

cost1= NN.costFunction(X,y)
#print djdW1
#print djdW2
scalar=100
NN.W1=NN.W1-scalar*djdW1
NN.W2=NN.W2-scalar*djdW2
cost2= NN.costFunction(X,y)

scalar=100*2
NN.W1=NN.W1+scalar*djdW1
NN.W2=NN.W2+scalar*djdW2
cost3= NN.costFunction(X,y)

print " original    "+str(cost1)
print " Added       "+str(cost3)
print " Subtracted  "+str(cost2)


class trainer(object):

    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))



    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 2000, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

NN=Neural_Network()
T=trainer(NN)
T.train(X,y)

print T.callbacksF

# import time
#
# weights=np.linspace(-5,5,1000)
# costs=np.zeros(1000)
#
#
# startTime=time.clock()
# for i in range(1000):
#     NN.W1[0,0]=weights[i]
#     yHat=NN.forward(X)
#     costs[i] = 0.5*sum((y-yHat)**2)
# endTime=time.clock()
#
#
# timeElapsed = endTime-startTime
# # print timeElapsed
# #
# # import matplotlib.pyplot as plt
# # plt.interactive(False)
# #
# # plt.plot(weights, costs)
# # plt.show()