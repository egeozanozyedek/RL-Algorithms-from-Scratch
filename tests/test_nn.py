 from src.NeuralNetwork.Network import Network
from src.NeuralNetwork.Layers import FullyConnected
import numpy as np
import matplotlib.pyplot as plt

arr = np.genfromtxt('../pima-indians-diabetes.csv', delimiter=',')
train = arr[:700]
test = arr[700:]
print(arr.shape)
X = train[:, :-1]
Y = train[:, -1].reshape(-1, 1)

X_t = test[:, :-1]
Y_t = test[:, -1].reshape(-1, 1)

print(X.shape, Y.shape)



layers = [FullyConnected(5, "tanh"), FullyConnected(5, "tanh"), FullyConnected(1, "sigmoid")]

net = Network(layers, X.shape[1], "MSE")
pred, loss = net.fit(X, Y, epoch=200, learning_rate=.4, momentum_rate=.3)

test_p = net.predict(X_t)

acc = (np.round(test_p) == Y_t).mean()

fig = plt.figure(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
fig.suptitle(f"Training MSE, Test Accuracy = {acc:.2f}")
plt.plot(loss, "C2")
plt.title("MSE")
plt.xlabel("Epoch")
# plt.show()
plt.savefig("../figures/report1/NN_test.png",  bbox_inches = 'tight')