#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import time

class RealTimeImshow():
    """
    Utility class for checking matrix iterations
    INPUT: initial 2D matrix

    METHODS:
        update_data(data) : replaces plot data
    """
    def __init__(self,init_data,cmap='gray',sleeptime=0.01):

        plt.figure()
        self.sleeptime = sleeptime
        self.ax = plt.subplot(1,1,1)
        self.img = plt.imshow(init_data,cmap=cmap)
        plt.show(block=False)
        plt.pause(0.001)

    def update_data(self,data):

        self.img.set_data(data)
        plt.pause(0.001)
        time.sleep(self.sleeptime)

# -------------------------TESTING----------------------------------------------

def main():

    def generate_data_():

        data = np.random.rand(100,100)
        return data


    init_data = generate_data_()
    rtplot = RealTimeImshow(init_data)
    for _ in range(100):
        data = generate_data_()
        rtplot.update_data(data)
        time.sleep(0.01)

if __name__ == '__main__':
    main()
