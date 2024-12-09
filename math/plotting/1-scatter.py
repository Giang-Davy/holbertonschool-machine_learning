#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def scatter():
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Plot the data as magenta points
    plt.scatter(x, y, color='magenta')

    # Set the x and y axis labels
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')

    # Set the title of the plot
    plt.title('Men\'s Height vs Weight')

    # Display the plot
    plt.show()
