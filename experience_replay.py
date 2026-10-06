# deque ek special data structure hai jo memory/buffer
# mein experiences ko store karne ke liye use hoga.
from collections import deque

# random module ka use memory se random experiences
# select karne ke liye hoga.
import random


# Ye class DQN ke liye Experience Replay Memory banayegi.
class ReplayMemory():

    # __init__() constructor hai.
    # Jab ReplayMemory ka object banega, ye automatically call hoga.
    #
    # maxlen = memory mein maximum kitni experiences store hongi
    # seed = randomization ke liye diya gaya parameter
    def __init__(self, maxlen, seed=None):

        # deque ke andar hum experiences store karenge.
        #
        # [] = initially memory empty hai.
        # maxlen = memory ki maximum capacity.
        #
        # Example:
        # maxlen = 10000
        # to memory mein maximum 10000 experiences hongi.
        self.memory = deque([], maxlen=maxlen)


    # Ye function ek new experience ko memory mein add karega.
    def append(self, new_exp):

        # new_exp ko replay memory ke end mein add kar rahe hain.
        #
        # Flappy Bird mein experience generally:
        # (state, action, reward, next_state, done)
        #
        # Example:
        # (state, 1, 0.1, next_state, False)
        self.memory.append(new_exp)


    # Ye function memory se randomly experiences select karega.
    #
    # sample_size = hume kitni experiences chahiye.
    def sample(self, sample_size):

        # random.sample() memory mein se random experiences choose karta hai.
        #
        # Example:
        # memory mein 1000 experiences hain
        # sample_size = 32
        #
        # to randomly 32 experiences milengi.
        return random.sample(self.memory, sample_size)


    # Ye function batata hai ki memory mein
    # currently kitni experiences stored hain.
    def __len__(self):

        # len(self.memory) memory ke andar
        # stored experiences ki total number return karega.
        return len(self.memory)