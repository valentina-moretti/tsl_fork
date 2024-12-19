import numpy as np
import matplotlib.pyplot as plt

def statsforecast_plot(len_input, len_pred, input_ts, y_hat, title, y_true=None):
    
    plt.figure(figsize=(30, 10))
    n_series = input_ts.shape[1]  
    print(input_ts.shape, y_hat.shape)
    colors = plt.cm.get_cmap("tab10", n_series) 
    
    for i in range(n_series):
        plt.plot(
            np.arange(len_input), 
            input_ts[:, i], 
            label=f'Input Series {i + 1}', 
            color=colors(i), 
            linestyle='-'
        )
        
        if y_true is not None:
            plt.plot(
                np.arange(len_input, len_input + len_pred), 
                y_true[:, i], 
                label=f'True Series {i + 1}', 
                color=colors(i), 
                linestyle='--'
            )
        plt.plot(
        np.arange(len_input, len_input + len_pred), 
        y_hat[:, i], 
        label=f'Forecast Series {i + 1}', 
        color=colors(i), 
        linestyle='-'
        )

    plt.title(title, fontsize=18)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'statsforecast_plot_{title.replace(" ", "_")}.png')
    plt.show()



def numpy_plot(len_pred, y_hat, title, y_true, input_ts=None, len_input=0):
    
    plt.figure(figsize=(30, 10))
    print(y_hat.shape)
    y_hat = y_hat[0].squeeze(1)
    y_true = y_true[0].squeeze(1)
    n_series = y_hat.shape[1]  
    colors = plt.cm.get_cmap("tab10", n_series) 
    
    for i in range(n_series):
        if input_ts is not None:
            plt.plot(
                np.arange(len_input), 
                input_ts[:, i], 
                label=f'Input Series {i + 1}', 
                color=colors(i), 
                linestyle='-'
            )
        
        if y_true is not None:
            plt.plot(
                np.arange(len_input, len_input + len_pred), 
                y_true[:, i], 
                label=f'True Series {i + 1}', 
                color=colors(i), 
                linestyle='--'
            )
        plt.plot(
        np.arange(len_input, len_input + len_pred), 
        y_hat[:, i], 
        label=f'Forecast Series {i + 1}', 
        color=colors(i), 
        linestyle='-'
        )

    plt.title(title, fontsize=18)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'statsforecast_plot_{title.replace(" ", "_")}.png')
    plt.show()
