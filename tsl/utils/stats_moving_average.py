import numpy as np

def stats_moving_average(input_array, kernel_size, stride=1):
    """
    Applies a moving average to the input data.

    :param input_array: NumPy array of shape (time_steps, nodes).
    :param kernel_size: Size of the moving average window.
    :param stride: Step size of the kernel.
    :return: NumPy array after applying the moving average, shape (output_length, nodes).
    """
    time_steps, nodes = input_array.shape
    
    # Pad just on the LEFT using the first value of the input array
    front = np.repeat(input_array[0:1, :], kernel_size - 1, axis=0)
    padded_array = np.concatenate([front, input_array], axis=0)
    
    output = np.zeros((len(input_array), nodes))
    
    for i in range(time_steps):
        start = i * stride
        end = start + kernel_size
        output[i, :] = np.mean(padded_array[start:end, :], axis=0)

    return output
