import numpy as np


def window_avg(numbers, window_size, threshold):
    i = 0
    size = numbers.shape[0]
    moving_averages = np.empty(size-window_size+1,)
    while i < size - window_size + 1:
        this_window = numbers[i: i + window_size]
        window_average = np.sum(this_window) / window_size
        moving_averages[i] = (window_average)
        i += 1

    return moving_averages, np.argmax(moving_averages > threshold)