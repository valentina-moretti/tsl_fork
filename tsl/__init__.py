from ._logger import logger
from .config import Config
from .lazy_loader import LazyLoader

data = LazyLoader('data', globals(), 'tsl.data')
datasets = LazyLoader('datasets', globals(), 'tsl.datasets')
nn = LazyLoader('nn', globals(), 'tsl.nn')
engines = LazyLoader('engines', globals(), 'tsl.engines')
utils = LazyLoader('utils', globals(), 'tsl.utils')

__version__ = '0.9.5'

epsilon = 5e-8
config = Config()

__all__ = [
    '__version__',
    'config',
    'epsilon',
    'logger',
    'data',
    'datasets',
    'nn',
    'engines',
    'utils',
]
