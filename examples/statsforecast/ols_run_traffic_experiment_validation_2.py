from typing import Optional

import torch
from omegaconf import DictConfig
from einops import rearrange
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
import numpy as np
from tsl.utils.plot import numpy_plot
import statsforecast.models as stat_models
from statsforecast.arima import arima_string
import matplotlib.pyplot as plt
from tsl.utils.plot import statsforecast_plot
import pandas as pd
from statsforecast import StatsForecast
from tsl import logger
from tsl.data.datamodule import SpatioTemporalDataModule
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.datasets.ltsf_benchmarks import ETTh1, ETTh2, ETTm1, ETTm2, ETTSplitter, LTSFSplitter, ElectricityDataset
from tsl.engines import Predictor
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.utils.casting import torch_to_numpy
from tsl.utils.stats_moving_average import stats_moving_average
from sklearn.linear_model import Ridge
import numpy as np
import random
import pytorch_lightning as pl
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
    elif dataset_name == 'electricity':
        dataset = ElectricityDataset(root='./tsl/.storage/electricity', drop_first=drop_first, dataset_name=cfg_dataset.name)

    else:
        raise ValueError(f"Dataset {dataset_name} not available.")
    return dataset


def run_traffic(cfg: DictConfig):
    
    pl.seed_everything(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    dataset = get_dataset(cfg.dataset.name, cfg.dataset)

    covariates = {'u': dataset.datetime_encoded(['hour', 'day']).values}
    
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

    transform = {'target': StandardScaler(axis=(0))}  # axis: time&space #TODO put axis=(0, 1) back


    if cfg.dataset.name in ['etth1', 'etth2', 'ettm1', 'ettm2']:
        splitter = ETTSplitter(dataset_name=cfg.dataset.name, seq_len=cfg.window, horizon=cfg.horizon)
    else:
        splitter = dataset.get_splitter(**cfg.dataset.splitting)

    print('head', torch_dataset.dataframe().head())
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=splitter,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
    )
    dm.setup()
    
    
    scaler = torch_dataset.scalers['target'].numpy()
    

    model = Ridge(alpha=cfg.model.hparams.alpha,
                    fit_intercept=cfg.model.hparams.fit_intercept,
                    copy_X=cfg.model.hparams.copy_X,
                    tol=cfg.model.hparams.tol,
                    solver=cfg.model.hparams.solver,
                    random_state=cfg.seed)
    
    train_window_idx = torch_dataset.get_window_indices(dm.trainset.indices)[:,0]
    val_window_idx = torch_dataset.get_window_indices(dm.valset.indices)[:,0]
    test_window_idx = torch_dataset.get_window_indices(dm.testset.indices)[:,0]
    
    if torch_dataset[train_window_idx].x.numpy().shape[2] == 1:
        squeeze_axis = 2
    else:
        squeeze_axis = 3
        
    train_window = torch_dataset[train_window_idx].x.numpy().squeeze(squeeze_axis) 
    val_window = torch_dataset[val_window_idx].x.numpy().squeeze(squeeze_axis)
    test_window = torch_dataset[test_window_idx].x.numpy().squeeze(squeeze_axis)
    
    train_horizon = scaler.transform(torch_dataset[train_window_idx].y.numpy()).squeeze(squeeze_axis)
    val_horizon = scaler.transform(torch_dataset[val_window_idx].y.numpy()).squeeze(squeeze_axis)
    test_horizon = scaler.transform(torch_dataset[test_window_idx].y.numpy()).squeeze(squeeze_axis)



    n_series = test_window.shape[2]
    if cfg.bool_individual == True:
        test_results_list = []
        val_result_list = []
        for j in range(n_series):
            train_window_j = train_window[:,:,j]
            val_window_j = val_window[:,:,j]
            test_window_j = test_window[:,:,j]

            train_horizon_j = train_horizon[:,:,j]
        

            model.fit(X=train_window_j, y=train_horizon_j) #TODO add covariates if possible
        
            val_pred_j = model.predict(val_window_j)
            test_pred_j = model.predict(test_window_j)
            
            val_result_list.append(val_pred_j)
            test_results_list.append(test_pred_j)
                
        val_result_array = np.stack(val_result_list, axis=2)
        test_results_array = np.stack(test_results_list, axis=2)

    else:
        x_fit = rearrange(train_window, 'b t f -> (b f) t')
        y_fit = rearrange(train_horizon, 'b t f -> (b f) t')
        model.fit(X=x_fit, y=y_fit)

        x_pred_val = rearrange(val_window, 'b t f -> (b f) t')
        x_pred_test = rearrange(test_window, 'b t f -> (b f) t')
        val_result_array = model.predict(x_pred_val)
        test_results_array = model.predict(x_pred_test)
        
        val_result_array = rearrange(val_result_array, '(b f) h -> b h f', f=n_series, h=cfg.horizon)
        test_results_array = rearrange(test_results_array, '(b f) h -> b h f', f=n_series, h=cfg.horizon)


        # working alternative
        # train_window = train_window.transpose(0, 2, 1)
        # train_horizon = train_horizon.transpose(0, 2, 1)
        # x_fit = train_window.reshape(-1, train_window.shape[-1])
        # y_fit = train_horizon.reshape(-1, train_horizon.shape[-1])
        # model.fit(X=x_fit, y=y_fit)
        # val_window = val_window.transpose(0, 2, 1)
        # test_window = test_window.transpose(0, 2, 1)
        # x_pred_val = val_window.reshape(-1, val_window.shape[-1])
        # x_pred_test = test_window.reshape(-1, test_window.shape[-1])
        # val_result_array = model.predict(x_pred_val).reshape(val_window.shape[0], val_window.shape[1], cfg.horizon)
        # test_results_array = model.predict(x_pred_test).reshape(test_window.shape[0], test_window.shape[1], cfg.horizon)
        # test_horizon = test_horizon.transpose(0, 2, 1)
        # val_horizon = val_horizon.transpose(0, 2, 1)

        

    print('test_results_array', test_results_array.shape, 'test_horizon', test_horizon.shape)
    print('val_result_array', val_result_array.shape, 'val_horizon', val_horizon.shape)
    
    #TODO remove comments
    # inverse the scaling
    # test_results_array = scaler.inverse_transform(test_results_array)
    # val_result_array = scaler.inverse_transform(val_result_array)
    # val_horizon = scaler.inverse_transform(val_true_array)
    # test_true_array = scaler.inverse_transform(test_true_array)

    
    print(f'MSE={np.mean((test_horizon-test_results_array)**2):0.3f}; MAE={np.mean(np.abs(test_horizon-test_results_array)):0.3f}.')
    print('\n--------------------- Results ---------------------\n')
    res  = dict(val_mae=numpy_metrics.mae(val_result_array, val_horizon),
                val_mse=numpy_metrics.mse(val_result_array, val_horizon),
                val_rmse=numpy_metrics.rmse(val_result_array, val_horizon),
                val_mape=numpy_metrics.mape(val_result_array, val_horizon),
                val_mase=numpy_metrics.mase(val_result_array, val_horizon),
                val_maape=numpy_metrics.maape(val_result_array, val_horizon),
                test_mae=numpy_metrics.mae(test_results_array, test_horizon),
                test_mse=numpy_metrics.mse(test_results_array, test_horizon),
                test_rmse=numpy_metrics.rmse(test_results_array, test_horizon),
                test_mape=numpy_metrics.mape(test_results_array, test_horizon),
                test_mase=numpy_metrics.mase(test_results_array, test_horizon),
                test_maape=numpy_metrics.maape(test_results_array, test_horizon)
                )
    
    return res


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic,
                     config_path='config/traffic',
                     config_name='default')
    res = exp.run()
    logger.info(res)
