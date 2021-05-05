import copy
import random
from collections import deque
import numpy as np


class ABC:
    def __init__(self, num):
        self.num = num
    def incrment(self):
        self.num += 1
        self.double()

    def double(self):
        print("sad")
        self.num *=2



class DBC(ABC):
    def __init__(self):
        super().__init__(1)
        self.target = copy.deepcopy(self)

    def aaa(self):
        super().incrment()

    def double(self):
        print("h")
        self.num += 2




a = DBC()

a.incrment()
print(a.num, a.target.num)


b = deque()
b.append([np.random.random(5), np.random.random(2), np.random.random(3)])
b.append((np.random.random(5), np.random.random(2), np.random.random(3)))
b.append((np.random.random(5), np.random.random(2), np.random.random(3)))
b.append((np.random.random(5), np.random.random(2), np.random.random(3)))

print(b)






sample = random.sample(b, 3)
states, actions, rewards = zip(*sample)
print(np.asarray(states).shape)



