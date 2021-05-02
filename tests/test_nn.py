import h5py

from src.NeuralNetwork.Network import Network
from src.NeuralNetwork.Layers import FullyConnected
import numpy as np
import matplotlib.pyplot as plt

# arr = np.genfromtxt('../assign2_data1.h5', delimiter=',')
# train = arr[:700]
# test = arr[700:]
# print(arr.shape)
# X = train[:, :-1]
# Y = train[:, -1].reshape(-1, 1)
#
# X_t = test[:, :-1]
# Y_t = test[:, -1].reshape(-1, 1)
#
# print(X.shape, Y.shape)


h5 = h5py.File("../assign2_data1.h5",'r')
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



layers = [FullyConnected(20, "tanh"), FullyConnected(5, "tanh"), FullyConnected(1, "tanh")]

net = Network(layers, X.shape[1], "MSE", optimizer="adam")
pred, loss = net.fit(X, Y, epoch=200, mini_batch_size=64, learning_rate=.0005)

test_p = net.predict(X_test)
acc = (np.sign(test_p) == Y_test).mean()

fig = plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
fig.suptitle(f"Training MSE, Test Accuracy = {acc:.2f}")
plt.plot(loss, "C2")
plt.title("MSE")
plt.xlabel("Epoch")
# plt.show()
plt.savefig("../figures/report1/NN_testsgd.png",  bbox_inches = 'tight')