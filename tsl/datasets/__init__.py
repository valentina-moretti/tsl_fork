# Interfaces
# isort: off
from .prototypes import Dataset, TabularDataset, DatetimeDataset
from .prototypes import classes as prototype_classes
# isort: on
# Datasets
from .air_quality import AirQuality
from .elergone import Elergone
from .gpvar import GPVARDataset, GPVARDatasetAZ
from .large_st import LargeST
from .metr_la import MetrLA
from .ltsf_benchmarks import ETTh1, ETTh2, ETTm1, ETTm2
from .mts_benchmarks import (ElectricityBenchmark, ExchangeBenchmark,
                             SolarBenchmark, TrafficBenchmark)
from .pems_bay import PemsBay
from .pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from .pv_us import PvUS
from .synthetic import GaussianNoiseSyntheticDataset

dataset_classes = [
    'AirQuality',
    'Elergone',
    'MetrLA',
    'PemsBay',
    'PeMS03',
    'PeMS04',
    'PeMS07',
    'PeMS08',
    'LargeST',
    'PvUS',
    'ElectricityBenchmark',
    'TrafficBenchmark',
    'SolarBenchmark',
    'ExchangeBenchmark',
    'GaussianNoiseSyntheticDataset',
    'GPVARDataset',
    'GPVARDatasetAZ',
    'ETTh1',
    'ETTh2',
    'ETTm1',
    'ETTm2',
    'ElectricityDataset'
]

__all__ = prototype_classes + dataset_classes
