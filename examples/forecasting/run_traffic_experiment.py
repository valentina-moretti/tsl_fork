from typing import Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
from tsl.utils.plot import numpy_plot
from tsl import logger
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.nn import models
from tsl.utils.casting import torch_to_numpy
import random
import pytorch_lightning as pl
import numpy as np
import gc
import urllib3
from tsl.engines import Predictor
from tsl.experiment import Experiment, NeptuneLogger
from tsl.datasets.prototypes.iid_dataset import IIDDataset
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.data.datamodule import SpatioTemporalDataModule
from tsl.datasets.ltsf_benchmarks import ETTh1, ETTh2, ETTm1, ETTm2, ETTSplitter, ElectricityDataset
from tsl.data.batch_map import BatchMapItem
from tsl.utils.timefeatures import time_features
# from tsl.utils import plot_lr_scheduler

def get_model_class(model_str):
    # Spatiotemporal Models ###################################################
    if model_str == 'dcrnn':
        model = models.DCRNNModel  # (Li et al., ICLR 2018)
    elif model_str == 'gwnet':
        model = models.GraphWaveNetModel  # (Wu et al., IJCAI 2019)
    elif model_str == 'evolvegcn':
        model = models.EvolveGCNModel  # (Pereja et al., AAAI 2020)
    elif model_str == 'agcrn':
        model = models.AGCRNModel  # (Bai et al., NeurIPS 2020)
    elif model_str == 'grugcn':
        model = models.GRUGCNModel  # (Guo et al., ICML 2022)
    elif model_str == 'gatedgn':
        model = models.GatedGraphNetworkModel  # (Satorras et al., 2022)
    elif model_str == 'stcn':
        model = models.STCNModel
    elif model_str == 'transformer':
        model = models.TransformerModel
    # Temporal Models #########################################################
    elif model_str == 'ar':
        model = models.ARModel
    elif model_str == 'var':
        model = models.VARModel
    elif model_str == 'dlinear':
        model = models.DLinearModel
    elif model_str == 'nlinear':
        model = models.NLinearModel
    elif model_str == 'rlinear':
        model = models.RLinearModel
    elif model_str == 'fits':
        model = models.FITSLinearModel
    elif model_str == 'es_trainable':
        model = models.ExponentialSmoothingModel
    elif model_str == 'rnn':
        model = models.RNNModel
    elif model_str == 'fcrnn':
        model = models.FCRNNModel
    elif model_str == 'tcn':
        model = models.TCNModel
    elif model_str == 'fctransformer':
        model = models.FCTransformerModel
    elif model_str == 'informer':
        model = models.InformerModel
    elif model_str == 'fcinformer':
        model = models.FCInformerModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name, drop_first=False, cfg_dataset=None): # TODO decide drop_first
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


def get_logger(cfg: DictConfig, exp: Experiment) -> Optional[Logger]:
    if cfg.logger is None:
        return None
    assert 'backend' in cfg.logger, \
        "cfg.logger must have a 'backend' attribute."
    if cfg.logger.backend == 'wandb':
        exp_logger = WandbLogger(name=cfg.run.name,
                                 save_dir=cfg.run.dir,
                                 offline=cfg.logger.offline,
                                 project=cfg.logger.project,
                                 config=exp.get_config_dict(),
                                 tags=cfg.tags)
    elif cfg.logger.backend == 'neptune':
        exp_logger = NeptuneLogger(project_name=cfg.logger.project,
                                   experiment_name=cfg.run.name,
                                   save_dir=cfg.run.dir,
                                   tags=cfg.tags,
                                   params=exp.get_config_dict(),
                                   debug=cfg.logger.offline,
                                   upload_stdout=False)
    elif cfg.logger.backend == 'tensorboard':
        exp_name = f'{cfg.run.name}_{"_".join(cfg.tags)}'
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=exp_name)
    else:
        raise ValueError(f"Logger {cfg.logger.backend} not available.")
    return exp_logger


def run_traffic(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    pl.seed_everything(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    dataset = get_dataset(dataset_name=cfg.dataset.name, cfg_dataset=cfg.dataset)

    bool_iid_dataset = cfg.bool_iid_dataset
    # encode time of the day and use it as exogenous variable
    # covariates = {'u': dataset.datetime_encoded(['hour', 'day']).values}
    if 'informer' in cfg.model.name:
        # datetime_idx is the sparse datetime_onehot encoding of the datetime, 
        # so instead of using the canonical onehot encoding (only one 1 in the row, eg. we need 24 columns for hours), 
        # we use the sparse onehot encoding (only 1 column for hours, with 24 values). 
        # In this way, we have 3 columns in total, one for the minute, one for the hour, and one for the day.
        # On the other hand, datetime_encoded produces two columns for each feature, one for the sin and one for the cos. 
        # So, in this case, we would have have 6 columns in total.
        # covariates = {'x_mark': dataset.datetime_idx(['month', 'day', 'weekday', 'hour']).values}
        timefeatures = time_features(dataset.dataframe().index, timeenc=1, freq='h')
        covariates = {'x_mark': timefeatures}
    else:
        covariates = None

    # get adjacency matrix
    adj = None
    # adj = dataset.get_connectivity(**cfg.dataset.connectivity)

    if cfg.bool_iid_dataset:
        torch_dataset = IIDDataset(
            target=dataset.dataframe(),
            mask=dataset.mask,
            connectivity=adj,
            covariates=covariates,
            horizon=cfg.horizon,
            window=cfg.window,
            stride=cfg.stride,
        )
        torch_dataset.set_batch_size(cfg.batch_size)
        torch_dataset.set_axis(cfg.axis)
    else:
        torch_dataset = SpatioTemporalDataset(
            target=dataset.dataframe(),
            mask=dataset.mask,
            connectivity=adj,
            covariates=covariates,
            horizon=cfg.horizon,
            window=cfg.window,
            stride=cfg.stride,
        )

    
    if 'informer' in cfg.model.name:
        # future_covariate = BatchMapItem( synch_mode= 'horizon', 
        #                                 pattern = 't f', t = 'time', f = 'feature')
        # torch_dataset.update_input_map('x_mark', future_covariate)
        torch_dataset.update_input_map(x_mark_horizon=('x_mark', 'horizon'))
        # torch_dataset.update_input_map(y_horizon=('target', 'horizon'))
    if cfg.axis:
        transform = {'target': StandardScaler(axis=(0))} 
    else:
        transform = {'target': StandardScaler(axis=(0,1))}  # axis: time&space #TODO rimettere axis=(0, 1), ho messo solo 0 perchè lo fa toner

    # TODO togliere splitter
    if cfg.dataset.name in ['etth1', 'etth2', 'ettm1', 'ettm2']:
        splitter = ETTSplitter(dataset_name=cfg.dataset.name, seq_len=cfg.window, horizon=cfg.horizon)
    else:
        splitter = dataset.get_splitter(**cfg.dataset.splitting)
    
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=splitter,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        bool_iid_dataset=bool_iid_dataset
    )
    dm.setup()

    ########################################
    # predictor                            #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    if covariates is None:
        exog_size = None 
    elif 'u' in torch_dataset.input_map.__dict__: 
        exog_size = torch_dataset.input_map.u.shape[-1]
    elif 'informer' in cfg.model.name:
        exog_size = torch_dataset.input_map.x_mark.shape[-1]
        print('exog_size', exog_size)
        print('covariates', covariates['x_mark'].shape)

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon,
                        window=torch_dataset.window,
                        exog_size= exog_size)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    loss_fn = torch_metrics.MaskedMSE()

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None
        
    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mape': torch_metrics.MaskedMAPE(),
        'mae_at_15': torch_metrics.MaskedMAE(at=2),  # 3rd is 15 min
        'mae_at_30': torch_metrics.MaskedMAE(at=5),  # 6th is 30 min
        'mae_at_60': torch_metrics.MaskedMAE(at=11),  # 12th is 1 h
        'mase': torch_metrics.MaskedMASE(),
        'maape': torch_metrics.MaskedMAAPE(),
    }

    predictor = Predictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=True,
    )
    

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mse',
        mode='min',
    )

    exp_logger = get_logger(cfg, exp)

    trainer = Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=cfg.run.dir,
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=cfg.grad_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=100, #100,
        limit_train_batches=cfg.train_batches,
    )
    trainer.fit(predictor, datamodule=dm)
    
    ########################################
    # testing                              #
    ########################################

    predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()
    trainer.test(predictor, datamodule=dm)

    output = trainer.predict(predictor, dataloaders=dm.test_dataloader())
    output = predictor.collate_prediction_outputs(output)
    output = torch_to_numpy(output)


        
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                        output.get('mask', None))
    scaler = torch_dataset.scalers['target'].numpy()
    
    y_hat = scaler.transform(y_hat)
    y_true = scaler.transform(y_true)
    
    res = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
            test_mse=numpy_metrics.mse(y_hat, y_true, mask),
            test_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
            test_mape=numpy_metrics.mape(y_hat, y_true, mask),
            test_mase=numpy_metrics.mase(y_hat, y_true, mask),
            test_maape=numpy_metrics.maape(y_hat, y_true, mask))

    # TODO put again when I restore scaling for test
    # output = trainer.predict(predictor, dataloaders=dm.val_dataloader())
    # output = predictor.collate_prediction_outputs(output)
    # output = torch_to_numpy(output)
    # y_hat, y_true, mask = (output['y_hat'], output['y'],
    #                     output.get('mask', None))
        
    # res.update(
    #     dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
    #         val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
    #         val_mape=numpy_metrics.mape(y_hat, y_true, mask),
    #         val_mase=numpy_metrics.mase(y_hat, y_true, mask),
    #         val_maape=numpy_metrics.maape(y_hat, y_true, mask))
    # )
   
    # numpy_plot(len_pred=cfg.horizon, y_hat=y_hat, title='Dlinear Forecasting', y_true=y_true)
    
    # plot_lr_scheduler(exp_logger, cfg.optimizer.hparams.lr, cfg.lr_scheduler.hparams.gamma, cfg.epochs)
    exp_logger.finalize('success')
    trainer.logger.finalize('success')
    exp_logger.experiment.stop()
    del predictor, trainer, dm, exp_logger, checkpoint_callback, early_stop_callback
    torch.cuda.empty_cache()
    gc.collect()
    urllib3.PoolManager().clear()
   
    return res['test_mae']


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic,
                     config_path='config/traffic',
                     config_name='search')
    res = exp.run()
    logger.info(res) 


