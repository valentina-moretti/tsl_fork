from typing import Optional

import torch
from omegaconf import DictConfig
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
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.datasets.ltsf_benchmarks import ETTh1, ETTh2, ETTm1, ETTm2, ETTSplitter 
from tsl.engines import Predictor
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.utils.casting import torch_to_numpy
from tsl.utils.stats_moving_average import stats_moving_average
from sklearn.linear_model import Ridge

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
    covariates = {'u': dataset.datetime_encoded(['hour', 'day']).values}
    # print('covariates', covariates)
    # print('dataset.datetime_encoded.values', covariates['u'])

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

    transform = {'target': StandardScaler(axis=(0))}  # axis: time&space #TODO put axis=(0, 1) back

    # TODO togliere il splitter
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
    scaled_dataset = scaler.transform(dataset.numpy())


    ########################################
    # predictor                            #
    ########################################

    model = Ridge(alpha=cfg.model.hparams.alpha,
                    fit_intercept=cfg.model.hparams.fit_intercept,
                    copy_X=cfg.model.hparams.copy_X,
                    tol=cfg.model.hparams.tol,
                    solver=cfg.model.hparams.solver,
                    random_state=cfg.seed)
    test_results_list = []
    val_true_list = []
    val_result_list = []
    test_true_list = []
    
    # train_target_indices_last = torch_dataset.get_horizon_indices(dm.valset.indices)[0,0].item()
    # val_target_indices_last = torch_dataset.get_horizon_indices(dm.testset.indices)[0,0].item()
    # # check
    # print(train_target_indices_last, '==', torch_dataset.get_window_indices(dm.trainset.indices)[-1,-1].item())
    # print(val_target_indices_last, '==', torch_dataset.get_window_indices(dm.valset.indices)[-1,-1].item())

    val_start = torch_dataset.get_window_indices(dm.valset.indices)[0,0].item()
    test_start = torch_dataset.get_window_indices(dm.testset.indices)[0,0].item()
    train_end = torch_dataset.get_horizon_indices(dm.trainset.indices)[-1,-1].item()
    val_end = torch_dataset.get_horizon_indices(dm.valset.indices)[-1,-1].item()
    test_end = torch_dataset.get_horizon_indices(dm.testset.indices)[-1,-1].item()

    print('val_start', val_start, 'test_start', test_start, 'train_end', train_end, 'val_end', val_end, 'test_end', test_end)
    
    
    
    assert train_end-cfg.horizon == torch_dataset.get_window_indices(dm.trainset.indices)[-1,-1].item()
    assert val_end-cfg.horizon == torch_dataset.get_window_indices(dm.valset.indices)[-1,-1].item()
    assert test_end-cfg.horizon == torch_dataset.get_window_indices(dm.testset.indices)[-1,-1].item()

    
    train_data = scaled_dataset[:train_end].squeeze(1)
    val_data = scaled_dataset[val_start:val_end].squeeze(1) #TODO METTERE- W ANCHE IN STATSFORCAST
    test_data = scaled_dataset[test_start:test_end+1].squeeze(1)

    train_dataframe = pd.DataFrame(train_data.reshape(-1, train_data.shape[-1]))
    print('train_dataframe', train_dataframe.describe())
    test_dataframe = pd.DataFrame(test_data.reshape(-1, test_data.shape[-1]))
    print('test_dataframe', test_dataframe.describe())

    if cfg.bool_mov_avg:
        train_data_m = stats_moving_average(input_array=train_data, kernel_size=cfg.mov_avg_window)
        val_data_m = stats_moving_average(input_array=val_data, kernel_size=cfg.mov_avg_window)
        test_data_m = stats_moving_average(input_array=test_data, kernel_size=cfg.mov_avg_window)
        train_data_r = train_data - train_data_m
        val_data_r = val_data - val_data_m
        test_data_r = test_data - test_data_m
        model_r = Ridge()
    
    print('train_data', train_data.shape)
    print('test_data', test_data.shape)

    n_series = test_data.shape[1]
    for j in range(n_series):
        val_true_j = val_data[:, j]
        
        if cfg.bool_mov_avg:
            train_data_r_j = train_data_r[:, j]
            train_data_m_j = train_data_m[:, j]
            val_data_r_j = val_data_r[:, j]
            val_data_m_j = val_data_m[:, j]

            test_data_r_j = test_data_r[:, j]
            test_data_m_j = test_data_m[:, j]

            train_window_r_j = np.array([train_data_r_j[i: i + cfg.window] for i in range(train_data_r_j.shape[0] - cfg.window - cfg.horizon + 1)])
            train_horizon_r_j = np.array([train_data_r_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(train_data_r_j.shape[0] - cfg.window - cfg.horizon + 1)])
            train_window_m_j = np.array([train_data_m_j[i: i + cfg.window] for i in range(train_data_m_j.shape[0] - cfg.window - cfg.horizon + 1)])
            train_horizon_m_j = np.array([train_data_m_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(train_data_m_j.shape[0] - cfg.window - cfg.horizon + 1)])

            val_window_r_j = np.array([val_data_r_j[i: i + cfg.window] for i in range(val_data_r_j.shape[0] - cfg.window - cfg.horizon + 1)])
            val_window_m_j = np.array([val_data_m_j[i: i + cfg.window] for i in range(val_data_m_j.shape[0] - cfg.window - cfg.horizon + 1)])
            test_window_r_j = np.array([test_data_r_j[i: i + cfg.window] for i in range(test_data_r_j.shape[0] - cfg.window - cfg.horizon + 1)])
            test_window_m_j = np.array([test_data_m_j[i: i + cfg.window] for i in range(test_data_m_j.shape[0] - cfg.window - cfg.horizon + 1)])
            
            model_r.fit(X=train_window_r_j, y=train_horizon_r_j) 
            val_pred_r = model_r.predict(val_window_r_j)
            test_pred_r = model_r.predict(test_window_r_j)

            model.fit(X=train_window_m_j, y=train_horizon_m_j)
            val_pred_m = model.predict(val_window_m_j)
            test_pred_m = model.predict(test_window_m_j)

            val_pred = val_pred_r + val_pred_m
            test_pred = test_pred_r + test_pred_m

            
            # train_window = np.concatenate([train_window_r_j, train_window_m_j], axis=1)
            # val_window = np.concatenate([val_window_r_j, val_window_m_j], axis=1)
            # test_window = np.concatenate([test_window_r_j, test_window_m_j], axis=1)
            # train_horizon = np.concatenate([train_horizon_r_j, train_horizon_m_j], axis=1)

            # model.fit(X=train_window, y=train_horizon)
            # val_pred_concat = model.predict(val_window)
            # test_pred_concat = model.predict(test_window)
            # val_pred = val_pred_concat[:, :val_window_r_j.shape[1]] + val_pred_concat[:, val_window_r_j.shape[1]:]
            # test_pred = test_pred_concat[:, :test_window_r_j.shape[1]] + test_pred_concat[:, test_window_r_j.shape[1]:]


            test_data_j = test_data[:, j]


        else:
            train_data_j = train_data[:, j]
            val_data_j = val_data[:, j]
            test_data_j = test_data[:, j]

            # create all the windows with the given window size and stride
            train_window_j = np.array([train_data_j[i: i + cfg.window] for i in range(train_data_j.shape[0] - cfg.window - cfg.horizon + 1)])
            val_window_j = np.array([val_data_j[i: i + cfg.window] for i in range(val_data_j.shape[0] - cfg.window - cfg.horizon + 1)])
            test_window_j = np.array([test_data_j[i: i + cfg.window] for i in range(test_data_j.shape[0] - cfg.window - cfg.horizon + 1)])

            # create the horizon
            train_horizon_j = np.array([train_data_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(train_data_j.shape[0] - cfg.window - cfg.horizon + 1)])
            val_horizon_j = np.array([val_data_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(val_data_j.shape[0] - cfg.window - cfg.horizon + 1)])
                        
            model.fit(X=train_window_j, y=train_horizon_j) #TODO add covariates if possible
        
            val_pred = model.predict(val_window_j)
            test_pred = model.predict(test_window_j)
        
        val_horizon_j = np.array([val_true_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(val_true_j.shape[0] - cfg.window - cfg.horizon + 1)])
        test_horizon_j = np.array([test_data_j[i + cfg.window: i + cfg.window + cfg.horizon] for i in range(test_data_j.shape[0] - cfg.window - cfg.horizon + 1)])
        test_results_list.append(test_pred)
        val_result_list.append(val_pred)
        val_true_list.append(val_horizon_j)
        test_true_list.append(test_horizon_j)
    
    
    test_results_array = np.stack(test_results_list, axis=2)
    val_result_array = np.stack(val_result_list, axis=2)
    val_true_array = np.stack(val_true_list, axis=2)
    test_true_array = np.stack(test_true_list, axis=2)

    print('val_result_array', val_result_array.shape)
    print('test_results_array', test_results_array.shape)
    #TODO remove comments
    # inverse the scaling
    # test_results_array = scaler.inverse_transform(test_results_array)
    # val_result_array = scaler.inverse_transform(val_result_array)
    # val_true_array = scaler.inverse_transform(val_true_array)
    # test_true_array = scaler.inverse_transform(test_true_array)

    print('\n--------------------- Results ---------------------\n')
    res  = dict(val_mae=numpy_metrics.mae(val_result_array, val_true_array),
                val_mse=numpy_metrics.mse(val_result_array, val_true_array),
                val_rmse=numpy_metrics.rmse(val_result_array, val_true_array),
                val_mape=numpy_metrics.mape(val_result_array, val_true_array),
                val_mase=numpy_metrics.mase(val_result_array, val_true_array),
                val_maape=numpy_metrics.maape(val_result_array, val_true_array),
                test_mae=numpy_metrics.mae(test_results_array, test_true_array),
                test_mse=numpy_metrics.mse(test_results_array, test_true_array),
                test_rmse=numpy_metrics.rmse(test_results_array, test_true_array),
                test_mape=numpy_metrics.mape(test_results_array, test_true_array),
                test_mase=numpy_metrics.mase(test_results_array, test_true_array),
                test_maape=numpy_metrics.maape(test_results_array, test_true_array)
                )
    
   
    numpy_plot(len_pred=cfg.horizon, y_hat=np.expand_dims(test_results_array, axis=2), title='OLS Dlinear', y_true=np.expand_dims(test_true_array, axis=2))
    
    return res


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic,
                     config_path='config/traffic',
                     config_name='default')
    res = exp.run()
    logger.info(res)
