from typing import Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
import numpy as np
import statsforecast.models as stat_models
from statsforecast.arima import arima_string
import matplotlib.pyplot as plt
from tsl.utils.plot import statsforecast_plot
import pandas as pd
from statsforecast import StatsForecast
from tsl import logger
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.datasets.ltsf_benchmarks import ETTh1, ETTh2, ETTm1, ETTm2
from tsl.engines import Predictor
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.utils.casting import torch_to_numpy
from tsl.utils.stats_moving_average import stats_moving_average


from concurrent.futures import ThreadPoolExecutor



def get_model_class(model_str, cfg, **kwargs):
    if model_str == 'autoarima':
        model = stat_models.AutoARIMA()
    elif model_str == 'arima':
        model = stat_models.ARIMA(order=(cfg.model.hparams.p, cfg.model.hparams.d, cfg.model.hparams.q))
    elif model_str == 'autoets':
        model = stat_models.AutoETS()
    elif model_str == 'autoces':
        model = stat_models.AutoCES()
    elif model_str == 'autotheta': # no model able to be fitted
        model = stat_models.AutoTheta()
    elif model_str == 'ses': 
        model = stat_models.SimpleExponentialSmoothing(alpha=cfg.model.hparams.alpha) # has no attribute 'forward'
    elif model_str == 'ses_optimized':
        model = stat_models.SimpleExponentialSmoothingOptimized()
    elif model_str == 'ses_sesonal':
        model = stat_models.SeasonalExponentialSmoothing()
    elif model_str == 'ses_sesonal_optimized':
        model = stat_models.SeasonalExponentialSmoothingOptimized()
    elif model_str == 'historicavg':
        model = stat_models.HistoricAverage() # has no attribute 'forward'
    elif model_str == 'naive':
        model = stat_models.Naive()
    elif model_str == 'random_drift':
        model = stat_models.RandomWalkWithDrift() # has no attribute 'forward'
    elif model_str == 'seasonal_naive':
        model = stat_models.SeasonalNaive()
    elif model_str == 'window_avg':
        model = stat_models.WindowAverage(window_size=cfg.model.hparams.window_size) # has no attribute 'forward'
    elif model_str == 'holt':
        model = stat_models.HoltWinters()
    elif model_str == 'mstl':
        model = stat_models.MSTL(season_length=[cfg.model.hparams.season_length ], trend_forecaster=stat_models.ARIMA(order=(cfg.model.hparams.p, cfg.model.hparams.d, cfg.model.hparams.q)))

    # ... add more models from https://github.com/Nixtla/statsforecast/blob/1e6d98c111cd33ec9ac45ab7e1b7b11b428e897b/python/statsforecast/models.py#L114
    # TODO those for many ts are here https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete.html
    else:
        raise ValueError(f"Model {model_str} not available.")
    return model

def get_dataset(dataset_name, cfg_dataset, drop_first=False):
    if dataset_name == 'la':
        dataset = MetrLA(impute_zeros=True)  # From Li et al. (ICLR 2018)
    elif dataset_name == 'bay':
        dataset = PemsBay()  # From Li et al. (ICLR 2018)
    elif dataset_name == 'pems3':
        dataset = PeMS03()  # From Guo et al. (2021)
    elif dataset_name == 'pems4':
        dataset = PeMS04()  # From Guo et al. (2021)
    elif dataset_name == 'pems7':
        dataset = PeMS07()  # From Guo et al. (2021)
    elif dataset_name == 'pems8':
        dataset = PeMS08()  # From Guo et al. (2021)
    elif dataset_name == 'etth1':
        dataset = ETTh1(root='../tsl/.storage/ETTh1', drop_first=drop_first, dataset_name=cfg_dataset.name)
    elif dataset_name == 'etth2':
        dataset = ETTh2(root='../tsl/.storage/ETTh1', drop_first=drop_first, dataset_name=cfg_dataset.name)
    elif dataset_name == 'ettm1':
        dataset = ETTm1(root='../tsl/.storage/ETTh1', drop_first=drop_first, dataset_name=cfg_dataset.name)
    elif dataset_name == 'ettm2':
        dataset = ETTm2(root='../tsl/.storage/ETTh1', drop_first=drop_first, dataset_name=cfg_dataset.name)
    else:
        raise ValueError(f"Dataset {dataset_name} not available.")
    return dataset


def run_traffic(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset.name, cfg.dataset)

    # encode time of the day and use it as exogenous variable
    covariates = {'u': dataset.datetime_encoded('hour').values}
    print('covariates', covariates)
    print('dataset.datetime_encoded.values', covariates['u'])

    # get adjacency matrix
    # adj = dataset.get_connectivity(**cfg.dataset.connectivity)

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        mask=dataset.mask,
        # connectivity=adj,
        covariates=covariates,
        horizon=cfg.horizon,
        window=cfg.window,
        stride=cfg.stride,
    )

    transform = {'target': StandardScaler(axis=(0, 1))}  # axis: time&space

    print('head', torch_dataset.dataframe().head())
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers,
    )
    dm.setup()
    
    scaler = torch_dataset.scalers['target'].numpy()
    scaled_dataset = scaler.transform(dataset.numpy())

    ########################################
    # predictor                            #
    ########################################

    model = get_model_class(cfg.model.name, cfg)
    

    # the model is fitted with the data to the last index of the validation set (of the horizon)
    val_target_indices_last = torch_dataset.get_horizon_indices(dm.testset.indices)[0,0].item()
    print('val_target_indices_last', val_target_indices_last)
    train_data_numpy = scaled_dataset[:val_target_indices_last]
    print('train_data_numpy' ,train_data_numpy.shape)
    not_scaled_data_numpy = dataset.numpy()[:val_target_indices_last].squeeze(1)

    train_data_numpy = train_data_numpy.squeeze(1)
    y_true_last = torch_dataset.get_horizon_indices(dm.testset.indices)[-1,-1].item()
    y_true = scaled_dataset[val_target_indices_last:y_true_last+1].squeeze(1)
    test_length = (y_true_last - (val_target_indices_last))
    print('y_true', y_true.shape)
    
    # print('last window val', torch_dataset.get_window_indices(dm.valset.indices)[-1,0], '-->', torch_dataset.get_window_indices(dm.valset.indices)[-1,-1])
    # print('last horizon val', torch_dataset.get_horizon_indices(dm.valset.indices)[-1,0], '-->', torch_dataset.get_horizon_indices(dm.valset.indices)[-1,-1])
    # print('first window test', torch_dataset.get_window_indices(dm.testset.indices)[0,0], '-->', torch_dataset.get_window_indices(dm.testset.indices)[0,-1])
    # print('first horizon test', torch_dataset.get_horizon_indices(dm.testset.indices)[0,0], '-->', torch_dataset.get_horizon_indices(dm.testset.indices)[0,-1])


    n_series = train_data_numpy.shape[1]
    n_samples = train_data_numpy.shape[0]
    
    results_list = []
    
    n_samples_trim = cfg.n_samples_trim
    
    results_list = [] 


    if cfg.bool_mov_avg:
        moving_average = stats_moving_average(input_array=train_data_numpy, kernel_size=cfg.mov_avg_window, stride=cfg.mov_avg_stride)
        print('moving_average', moving_average.shape)
        residuals = train_data_numpy - moving_average
        y_true_m = stats_moving_average(input_array=y_true, kernel_size=cfg.mov_avg_window, stride=cfg.mov_avg_stride)
        y_true_r = y_true - y_true_m

        

    for j in range(n_series):
        print('j', j)
        # fit one model per each time series independently on the training data

        if cfg.bool_mov_avg:
            input_ts_r = residuals[:, j]
            input_ts_m = moving_average[:, j]
            model_r = get_model_class(cfg.model.name, cfg)
            model.fit(input_ts_m)
            model_r.fit(input_ts_r)

        else:
            input_ts = train_data_numpy[:, j]
            if cfg.bool_exog:
                res = model.fit(input_ts, X=covariates['u'][:val_target_indices_last])
            else:
                result = model.fit(input_ts)
            # print parameters of the model
            if cfg.model.name == 'autoarima' or cfg.model.name == 'arima':
                print('arima params:', arima_string(model.model_))

        
        len_loop = test_length - cfg.horizon + 1
        
        for i in range(0, len_loop): 
            if cfg.bool_mov_avg:
                input_ts_m = input_ts_m[-n_samples_trim:]  
                input_ts_r = input_ts_r[-n_samples_trim:]  
                y_hat_m = model.forward(input_ts_m, h=cfg.horizon, fitted=True)['mean'] 
                y_hat_r = model_r.forward(input_ts_r, h=cfg.horizon, fitted=True)['mean'] 
                y_hat = y_hat_m + y_hat_r
                input_ts_m = np.concatenate((input_ts_m, y_true_m[i:i+1, j]), axis=0)
                input_ts_r = np.concatenate((input_ts_r, y_true_r[i:i+1, j]), axis=0)
            else:
                input_ts = input_ts[-n_samples_trim:]  
                # input to the forward is the true values seen preceding the prediction, the model is not fitted on them
                if cfg.bool_exog:
                    cov = covariates['u'][val_target_indices_last+i-n_samples_trim:val_target_indices_last+i]
                    cov_future = covariates['u'][val_target_indices_last+i:val_target_indices_last+i+cfg.horizon]
                    y_hat = model.forward(y=input_ts, h=cfg.horizon, fitted=False, X=cov, X_future=cov_future)['mean']
                else:
                    y_hat = model.forward(y=input_ts, h=cfg.horizon, fitted=False)['mean'] 
                input_ts = np.concatenate((input_ts, y_true[i:i+1, j]), axis=0)  
                # trim the input_ts to the last n_samples_trim because we don't need to keep all the history
                

            # add the prediction to the results, along the number of predictions made
            y_hat = y_hat[np.newaxis, :] 
            if i == 0:
                result = y_hat
            else:
                result = np.concatenate((result, y_hat), axis=0)  
            
        # add the list of predictions for each time series
        results_list.append(result) 
    
    results_array = np.stack(results_list, axis=-1)  
    # statsforecast_plot(len_input=500, len_pred=cfg.horizon, input_ts=train_data_numpy[-500:], y_hat=results_array[0], title=cfg.model.name+'_scaled')


    # generate all y_true with sliding window = 1 and horizon = cfg.horizon
    y_true_list = []
    for i in range(0, test_length-cfg.horizon+1):
        y_true_list.append(y_true[i:i+cfg.horizon])
    y_true = np.stack(y_true_list, axis=0)  
    
    print('result_all_ts', results_array.shape, 'y_true', y_true.shape)
    
    

    # Inverse scaling of the predictions and the true values
    results_array = scaler.inverse_transform(results_array)
    y_true = scaler.inverse_transform(y_true)
    print('result_all_ts', results_array.shape, 'y_true', y_true.shape, 'not_scaled_data_numpy', not_scaled_data_numpy.shape)
    # TODO check features last dimension
    
    # statsforecast_plot(len_input=500, len_pred=cfg.horizon, input_ts=not_scaled_data_numpy[-500:], y_hat=results_array[0], title=cfg.model.name, y_true=y_true[0])

    res = dict(test_mae=numpy_metrics.mae(results_array, y_true),
               test_mse=numpy_metrics.mse(results_array, y_true),
               test_rmse=numpy_metrics.rmse(results_array, y_true),
               test_mape=numpy_metrics.mape(results_array, y_true),
               test_mase=numpy_metrics.mase(results_array, y_true),
               test_maape=numpy_metrics.maape(results_array, y_true)
                )
    
    
    return res


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic,
                     config_path='config/traffic',
                     config_name='default')
    res = exp.run()
    logger.info(res)
