import h5py
from src.NeuralNetwork.Network import Network
from src.NeuralNetwork.Layers import FullyConnected
import numpy as np
import matplotlib.pyplot as plt



filename = "../assign2_data1.h5"
h5 = h5py.File(filename,'r')
trainims = h5['trainims'][()].astype('float64').transpose(0, 2, 1)
trainlbls = h5['trainlbls'][()].astype(int)
testims = h5['testims'][()].astype('float64').transpose(0, 2, 1)
testlbls = h5['testlbls'][()].astype(int)
h5.close()

trainlbls[np.where(trainlbls == 0)] = -1
testlbls[np.where(testlbls == 0)] = -1

X = np.reshape(trainims, (trainims.shape[0], trainims.shape[1] * trainims.shape[2]))
X = 1 * X/np.amax(X)
Y = np.reshape(trainlbls, (trainlbls.shape[0], 1))
X_test = np.reshape(testims, (testims.shape[0], testims.shape[1] * testims.shape[2]))
X_test = 1 * X_test/np.amax(X_test)
Y_test = np.reshape(testlbls, (testlbls.shape[0], 1))

# Y[Y == -1] = 0
print(X.shape, Y.min(), Y.max())


layers = [FullyConnected(20, "tanh"), FullyConnected(5, "tanh"), FullyConnected(1, "tanh")]

net = Network(layers, X.shape[1], "MSE")
pred, loss = net.fit(X, Y, epoch=300, learning_rate=.3, momentum_rate=.1, batch_size=50)
# acc = (np.sign(pred) == Y).mean() * 100

# print(acc)

fig = plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
fig.suptitle("Testing")
plt.plot(loss, "C3")
plt.title("MSE")
plt.xlabel("Epoch")
plt.show()