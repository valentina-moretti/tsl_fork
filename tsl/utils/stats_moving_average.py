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
    
    # Symmetric padding on both ends of the time dimension
    pad_width = (kernel_size - 1) // 2
    front = np.repeat(input_array[:1, :], pad_width, axis=0)
    end = np.repeat(input_array[-1:, :], pad_width, axis=0)
    padded_array = np.concatenate([front, input_array, end], axis=0)
    
    output_length = (time_steps + 2 * pad_width - kernel_size) // stride + 1
    output = np.zeros((output_length, nodes))
    
    for i in range(output_length):
        start = i * stride
        end = start + kernel_size
        output[i, :] = np.mean(padded_array[start:end, :], axis=0)

    return output
