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
from sklearn.linear_model import LinearRegression

from concurrent.futures import ThreadPoolExecutor



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

    model = LinearRegression()
    results_list = []
    y_true_list = []
    
    
    val_target_indices_last = torch_dataset.get_horizon_indices(dm.testset.indices)[0,0].item()
    train_data_numpy = scaled_dataset[:val_target_indices_last]

    train_data_numpy = train_data_numpy.squeeze(1)
    y_true_last = torch_dataset.get_horizon_indices(dm.testset.indices)[-1,-1].item()
    y_true = scaled_dataset[val_target_indices_last:y_true_last+1].squeeze(1)

    if cfg.bool_mov_avg:
        train_data_numpy_m = stats_moving_average(input_array=train_data_numpy, kernel_size=cfg.mov_avg_window)
        y_true_m = stats_moving_average(input_array=y_true, kernel_size=cfg.mov_avg_window)
        train_data_numpy_r = train_data_numpy - train_data_numpy_m
        y_true_r = y_true - y_true_m
        model_r = LinearRegression()
    
    print('train_data_numpy', train_data_numpy.shape)
    print('y_true', y_true.shape)

    n_series = y_true.shape[1]
    for j in range(n_series):
        
        if cfg.bool_mov_avg:
            train_data_numpy_r_j = train_data_numpy_r[:, j]
            train_data_numpy_m_j = train_data_numpy_m[:, j]
            y_true_r_j = y_true_r[:, j]
            y_true_m_j = y_true_m[:, j]

            train_window_r_j = np.array([train_data_numpy_r_j[i: i + cfg.window] for i in range(train_data_numpy_r_j.shape[0] - cfg.window - cfg.horizon + 1)])
            train_horizon_r_j = np.array([train_data_numpy_r_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(train_data_numpy_r_j.shape[0] - cfg.window - cfg.horizon + 1)])
            train_window_m_j = np.array([train_data_numpy_m_j[i: i + cfg.window] for i in range(train_data_numpy_m_j.shape[0] - cfg.window - cfg.horizon + 1)])
            train_horizon_m_j = np.array([train_data_numpy_m_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(train_data_numpy_m_j.shape[0] - cfg.window - cfg.horizon + 1)])

            test_window_r_j = np.array([y_true_r_j[i: i + cfg.window] for i in range(y_true_r_j.shape[0] - cfg.window - cfg.horizon + 1)])
            test_window_m_j = np.array([y_true_m_j[i: i + cfg.window] for i in range(y_true_m_j.shape[0] - cfg.window - cfg.horizon + 1)])
            
            model.fit(X=train_window_r_j, y=train_horizon_r_j) 
            y_pred_r = model.predict(test_window_r_j)
            model_r.fit(X=train_window_m_j, y=train_horizon_m_j)
            y_pred_m = model_r.predict(test_window_m_j)

            y_pred = y_pred_r + y_pred_m

            y_true_j = y_true[:, j]


        else:
            train_data_numpy_j = train_data_numpy[:, j]
            y_true_j = y_true[:, j]

            # create all the windows with the given window size and stride
            train_window_j = np.array([train_data_numpy_j[i: i + cfg.window] for i in range(train_data_numpy_j.shape[0] - cfg.window - cfg.horizon + 1)])

            # create the horizon
            train_horizon_j = np.array([train_data_numpy_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(train_data_numpy_j.shape[0] - cfg.window - cfg.horizon + 1)])
            
            model.fit(X=train_window_j, y=train_horizon_j) #TODO add covariates if possible

            # create the test window
            test_window_j = np.array([y_true_j[i: i + cfg.window] for i in range(y_true_j.shape[0] - cfg.window - cfg.horizon + 1)])
        
            y_pred = model.predict(test_window_j)
        
        test_horizon_j = np.array([y_true_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(y_true_j.shape[0] - cfg.window - cfg.horizon + 1)])
        results_list.append(y_pred)
        y_true_list.append(test_horizon_j)
    
    
    results_array = np.stack(results_list, axis=2)
    y_true_array = np.stack(y_true_list, axis=2)

    # inverse the scaling
    results_array = scaler.inverse_transform(results_array)
    y_true_array = scaler.inverse_transform(y_true_array)

    
    res = dict(test_mae=numpy_metrics.mae(results_array, y_true_array),
               test_mse=numpy_metrics.mse(results_array, y_true_array),
               test_rmse=numpy_metrics.rmse(results_array, y_true_array),
               test_mape=numpy_metrics.mape(results_array, y_true_array),
               test_mase=numpy_metrics.mase(results_array, y_true_array),
               test_maape=numpy_metrics.maape(results_array, y_true_array)
                )
    
    
    return res


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic,
                     config_path='config/traffic',
                     config_name='default')
    res = exp.run()
    logger.info(res)
