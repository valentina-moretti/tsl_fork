from typing import Optional
import pandas as pd
import torch
import numpy as np
from tsl.data import Splitter, Data

from tsl.datasets.prototypes import DatetimeDataset
from tsl.utils import download_url
from tsl.data.preprocessing import ScalerModule
from tsl.data import AtTimeStepSplitter

# MONASH
from datetime import datetime
from distutils.util import strtobool
import pandas as pd
import matplotlib.pyplot as plt
import h5py

###################   DATASETS   ###################

class _LTSFDataset(DatetimeDataset):
    url = None
    default_freq = None
    similarity_options = None

    def __init__(self, drop_first, dataset_name, root=None, freq=None):
        self.root = root
        self.drop_first = drop_first
        self.dataset_name = dataset_name
        df = self.load()
        super().__init__(target=df,
                         freq=freq,
                         temporal_aggregation='mean',
                         spatial_aggregation='sum',
                         name=self.__class__.__name__)

    @property
    def required_file_names(self):
        return [f'{self.__class__.__name__}.h5']
    

    def download(self) -> None:
        assert self.url is not None, "You must specify the url of the dataset"
        assert len(self.raw_file_names), "You must specify the raw file names"
        print("Downloading from ", self.url, " to ", self.root_dir)
        download_url(self.url, self.root_dir, self.raw_file_names[0])
    

    # Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
    #
    # Parameters
    # full_file_path_and_name - complete .tsf file path
    # replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
    # value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
    


    # Example of usage
    # loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("TSForecasting/tsf_data/sample.tsf")

    # print(loaded_data)
    # print(frequency)
    # print(forecast_horizon)
    # print(contain_missing_values)
    # print(contain_equal_length)


    def read_custom_format(self) -> pd.DataFrame:
        file_path = self.raw_files_paths[0]
        print(file_path)

        # if file ends with .p, we read it as a pickle file
        if file_path.endswith('.p'):
            df = pd.read_pickle(file_path)
            '''
            #for exchange tar.gz with link
            if file_path.endswith('.txt.gz'):
                df = pd.read_csv(file_path, delimiter=',', decimal='.', header=None, compression='gzip')
                # add temporal labels
                date_range = pd.date_range(start='1990-01-01', periods=len(df), freq='1d')
                df.index = pd.to_datetime(date_range, format='%Y-%m-%d %H:%M')
            '''
        elif file_path.endswith('.tsf'):
            print(file_path)
            full_file_path_and_name=file_path
            replace_missing_vals_with="NaN"
            value_column_name="series_value"
            
            col_names = []
            col_types = []
            all_data = {}
            line_count = 0
            frequency = None
            forecast_horizon = None
            contain_missing_values = None
            contain_equal_length = None
            found_data_tag = False
            found_data_section = False
            started_reading_data_section = False
            print(full_file_path_and_name)
            with open(full_file_path_and_name, "r", encoding="cp1252") as file:
                for line in file:
                    # Strip white space from start/end of line
                    line = line.strip()

                    if line:
                        if line.startswith("@"):  # Read meta-data
                            if not line.startswith("@data"):
                                line_content = line.split(" ")
                                if line.startswith("@attribute"):
                                    if (
                                        len(line_content) != 3
                                    ):  # Attributes have both name and type
                                        raise Exception("Invalid meta-data specification.")

                                    col_names.append(line_content[1])
                                    col_types.append(line_content[2])
                                else:
                                    if (
                                        len(line_content) != 2
                                    ):  # Other meta-data have only values
                                        raise Exception("Invalid meta-data specification.")

                                    if line.startswith("@frequency"):
                                        frequency = line_content[1]
                                    elif line.startswith("@horizon"):
                                        forecast_horizon = int(line_content[1])
                                    elif line.startswith("@missing"):
                                        contain_missing_values = bool(
                                            strtobool(line_content[1])
                                        )
                                    elif line.startswith("@equallength"):
                                        contain_equal_length = bool(strtobool(line_content[1]))

                            else:
                                if len(col_names) == 0:
                                    raise Exception(
                                        "Missing attribute section. Attribute section must come before data."
                                    )

                                found_data_tag = True
                        elif not line.startswith("#"):
                            if len(col_names) == 0:
                                raise Exception(
                                    "Missing attribute section. Attribute section must come before data."
                                )
                            elif not found_data_tag:
                                raise Exception("Missing @data tag.")
                            else:
                                if not started_reading_data_section:
                                    started_reading_data_section = True
                                    found_data_section = True
                                    all_series = []

                                    for col in col_names:
                                        all_data[col] = []

                                full_info = line.split(":")

                                if len(full_info) != (len(col_names) + 1):
                                    raise Exception("Missing attributes/values in series.")

                                series = full_info[len(full_info) - 1]
                                series = series.split(",")

                                if len(series) == 0:
                                    raise Exception(
                                        "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                                    )

                                numeric_series = []

                                for val in series:
                                    if val == "?":
                                        numeric_series.append(replace_missing_vals_with)
                                    else:
                                        numeric_series.append(float(val))

                                if numeric_series.count(replace_missing_vals_with) == len(
                                    numeric_series
                                ):
                                    raise Exception(
                                        "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                                    )

                                all_series.append(numeric_series)

                                for i in range(len(col_names)):
                                    att_val = None
                                    if col_types[i] == "numeric":
                                        att_val = int(full_info[i])
                                    elif col_types[i] == "string":
                                        att_val = str(full_info[i])
                                    elif col_types[i] == "date":
                                        att_val = datetime.strptime(
                                            full_info[i], "%Y-%m-%d %H-%M-%S"
                                        )
                                    else:
                                        raise Exception(
                                            "Invalid attribute type."
                                        )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                                    if att_val is None:
                                        raise Exception("Invalid attribute value.")
                                    else:
                                        all_data[col_names[i]].append(att_val)

                        line_count = line_count + 1

                if line_count == 0:
                    raise Exception("Empty file.")
                if len(col_names) == 0:
                    raise Exception("Missing attribute section.")
                if not found_data_section:
                    raise Exception("Missing series information under data section.")

                all_data[value_column_name] = all_series
                loaded_data = pd.DataFrame(all_data)
                
                new_data = pd.DataFrame()
                
                print(loaded_data.head())

                if self.dataset_name == 'M4':

                    all_data[value_column_name] = all_series
                    loaded_data = pd.DataFrame(all_data)

                    # Trova la lunghezza massima tra tutte le serie temporali
                    max_len = max(len(series) for series in loaded_data["series_value"])
                    
                
                for i, el in enumerate(loaded_data["series_name"]):
                    series = loaded_data.iloc[i]["series_value"]
                    if self.dataset_name == 'M4' and len(series) < max_len:
                            series = series + [np.nan] * (max_len - len(series))
                        
                    new_data[el] = series
                    
                len_series = max_len if self.dataset_name == 'M4' else len(loaded_data.iloc[0]["series_value"])
                index = pd.date_range(start=loaded_data.iloc[0]["start_timestamp"], periods=len_series, freq=self.default_freq)
                print(len(loaded_data))
                print("len series: ", len_series)
                new_data = new_data.set_index(index)
                
                print(new_data.head())
                
                df = new_data
                

                # return (
                #     loaded_data,
                #     frequency,
                #     forecast_horizon,
                #     contain_missing_values,
                #     contain_equal_length,
                # )
            print("*******************************")
            print(loaded_data.head())
            print("Frequency: ", frequency)
            print("Forecast horizon: ", forecast_horizon)
            print("Contain Missing values: ", contain_missing_values)
            print("Contain_equal_length: ", contain_equal_length)
            print("*******************************")
            plt.figure(figsize=(20, 10))
            for i in range(loaded_data.shape[0]):
                #print(loaded_data.head())
                plt.plot(loaded_data.iloc[i]["series_value"])
            plt.title(file_path)
            plt.tight_layout()  # Aggiusta automaticamente la disposizione delle subplot per evitare sovrapposizioni
            plt.savefig(file_path.split(".")[0] + "_plot.png")
            plt.close()
                        

        else:
            df = pd.read_csv(file_path, delimiter=',', index_col=0, parse_dates=True)
            
        # print df precision
        print("SHAPE: ", df.shape)
        print("TYPES: ", df.dtypes)
        # find if there are na
        print("NA: ", df.isna().sum().sum())

        # fill na
        df.fillna(method='ffill', inplace=True, axis=0)

        
        print("TYPES: ", df.dtypes)
        print("SHAPE: ", df.shape)
        # float64_32_cols = list(df.select_dtypes(include=['float64', 'float32']).columns)

        # # The same code again calling the columns
        # df[float64_32_cols] = df[float64_32_cols].astype('float16')

        print("NA: ", df.isna().sum().sum())

        # df.index = pd.to_datetime(df.index)  # convert index to datetime
        index = pd.date_range(start=df.index[0],
                              periods=len(df),
                              freq=self.default_freq)
        df = df.set_index(index)
        
        # print(df.head())
        
        if self.default_freq is not None:
            df = df.asfreq(self.default_freq)  # resample to default frequency
        
        # check if it has na
        if df.isna().sum().sum() > 0:
            # print rows with na
            print(df[df.isna().any(axis=1)])
        else:
            print("No missing values")
        
        # Change columns to multiindex
        
        if not self.bool_multinode:
            df.columns = pd.MultiIndex.from_product([['global_node'], df.columns],
                                                    names=['node', 'channel'])

        print(df)
        return df

    def build(self):
        # Build dataset
        print(self.raw_file_names)
        self.maybe_download()
        print(f"Building the {self.__class__.__name__} dataset...")
        df = self.read_custom_format()
        print(df.shape, 'shape data')
        filename = self.required_files_paths[0]
        df.to_hdf(filename, key='data', mode='w', complevel=3)
        self.clean_downloads()

    def load_raw(self) -> pd.DataFrame:
        self.maybe_build()
        df = pd.read_hdf(self.required_files_paths[0], key='data')
        #print("Load raw finished")
        # print(df.head())
        print("Dataset shape before drop: ", df.shape)
        # print("Df Index: ", df.index)
        print("NA: ", df.isna().sum().sum())
        

        # if we are not using the lagged data and we are not using echo
        if self.drop_first:
            df_new = df.iloc[1:,:]
        # if we are not using the lagged data and we are using echo
        else:
            df_new = df

        # df.ffill(inplace=True, axis=0)
        df_new.fillna(method='ffill', inplace=True, axis=0)
        
        # convert to 16 bits to save space if it's not M4
        # df_new = df_new.astype('float16') if self.dataset_name != 'M4' and self.dataset_name != 'electricity'  else df_new

        # remove second feature 
        # df_new = df_new.drop(df_new.columns[1], axis=1)
        
        return df_new

    def get_splitter(self, name_dataset, seed: int = 41,  method: Optional[str] = None,
                     seq_len: int = 336, horizon: int = 720, **kwargs) -> Splitter:
        name = self.__class__.__name__
        name = name.replace('Dataset', '')
        if name_dataset == 'ETTh1' or name_dataset == 'ETTh2' or name_dataset == 'ETTm1' or name_dataset == 'ETTm2':
            return ETTSplitter(dataset=name_dataset, seq_len=seq_len, horizon= horizon)
        else:
            if name_dataset == 'metr_la':
                return AtTimeStepSplitter(first_val_ts=(2012, 5, 25, 16, 00),
                                        last_val_ts=(2012, 6, 4, 3, 20),
                                        first_test_ts=(2012, 6, 4, 4, 20))
            elif name_dataset == 'pems_bay':
                return AtTimeStepSplitter(first_val_ts=(2017, 5, 11, 7, 20),
                                        last_val_ts=(2017, 5, 25, 17, 40),
                                        first_test_ts=(2017, 5, 25, 18, 40))
            elif name_dataset == 'traffic':
                return AtTimeStepSplitter(first_val_ts=(2016, 6, 9, 23, 00, 1),
                                        last_val_ts=(2016, 8, 6, 6, 00, 1),
                                        first_test_ts=(2016, 8, 7, 6, 00, 1))
            
            elif name_dataset == 'electricity':
                return AtTimeStepSplitter(first_val_ts=(2014, 2, 27, 18, 00, 1),
                                        last_val_ts=(2014, 5, 25, 6, 00, 1),
                                        first_test_ts=(2014, 5, 26, 6, 00, 1))
            elif name_dataset == 'exchange':
                return AtTimeStepSplitter(first_val_ts=(2004, 4, 11, 00, 00),
                                        last_val_ts=(2005, 10, 9, 00, 00),
                                        first_test_ts=(2005, 11, 8, 00, 00))
            elif name_dataset == 'M4':
                return AtTimeStepSplitter(first_val_ts=(2015, 7, 31, 9, 00),
                                        last_val_ts=(2015, 8, 2, 13, 00),
                                        first_test_ts=(2015, 8, 3, 13, 00))
            elif name_dataset == 'elergone':
                return AtTimeStepSplitter(first_val_ts=(1972, 2, 28, 21, 00),
                                        last_val_ts=(1972, 5, 26, 13, 00),
                                        first_test_ts=(1972, 5, 27, 13, 00))

class ETTh1(_LTSFDataset):
    url = ('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/'
           'main/ETT-small/ETTh1.csv')
    default_freq = '1h'
    bool_multinode = False

    @property
    def raw_file_names(self):
        return ['ETTh1.csv']


class ETTh2(_LTSFDataset):
    url = ('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/'
           'main/ETT-small/ETTh2.csv')
    default_freq = '1h'
    bool_multinode = False

    @property
    def raw_file_names(self):
        return ['ETTh2.csv']


class ETTm1(_LTSFDataset):
    url = ('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/'
           'main/ETT-small/ETTm1.csv')
    default_freq = '15min'
    bool_multinode = False

    @property
    def raw_file_names(self):
        return ['ETTm1.csv']


class ETTm2(_LTSFDataset):
    url = ('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv')
    default_freq = '15min'
    bool_multinode = False

    @property
    def raw_file_names(self):
        return ['ETTm2.csv']
    
class TrafficDataset(_LTSFDataset):
    url = ('https://drive.google.com/file/d/1U3BZ3Wvuvd9HVAx5Nl3bHYG9rsh5-yZX/view?usp=sharing') #TODO change link
    default_freq = '1h'
    bool_multinode = True

    @property
    def raw_file_names(self):
        return ['traffic_hourly_dataset.tsf']    
    
    
class PemsBayDataset(_LTSFDataset):
    url = None #TODO change link
    default_freq = '5min'
    bool_multinode = True

    @property
    def raw_file_names(self):
        return ['pems_bay.csv']
    
class MetrLaDataset(_LTSFDataset):
    url = None #TODO change link
    default_freq = '5min'
    bool_multinode = True

    @property
    def raw_file_names(self):
        return ['METR-LA.csv'] 
    
class M4Dataset(_LTSFDataset):
    url = ('https://drive.google.com/file/d/1U3BZ3Wvuvd9HVAx5Nl3bHYG9rsh5-yZX/view?usp=sharing') #TODO change link
    default_freq = '1h'
    bool_multinode = True

    @property
    def raw_file_names(self):
        return ['m4_hourly_dataset.tsf']
    
class ElectricityDataset(_LTSFDataset):
    url = None #('https://drive.google.com/file/d/1U3BZ3Wvuvd9HVAx5Nl3bHYG9rsh5-yZX/view?usp=sharing') #TODO change link
    default_freq = '1h'
    bool_multinode = True

    @property
    def raw_file_names(self):
        return ['electricity_hourly_dataset.tsf']
    

class ExchangeRateDataset(_LTSFDataset):
    # https://github.com/laiguokun/multivariate-time-series-data/tree/master?tab=readme-ov-file
    # reference paper: https://arxiv.org/pdf/1703.07015
    url = ('https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz')
    default_freq = '1d'
    bool_multinode = False

    @property
    def raw_file_names(self):
        return ['exchange.csv']


class Elergone(_LTSFDataset):
    """Electricity consumption (in kWh) measured hourly by 321 sensors from
    2012 to 2014.

    Imported from https://github.com/laiguokun/multivariate-time-series-data.
    The `original dataset <https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014>`_
    records values in kW for 370 nodes starting from 2011, with part of the
    nodes with missing values before 2012. For the original dataset refer to
    :class:`~tsl.datasets.Elergone`.

    Dataset information:
        + Time steps: 26304
        + Nodes: 321
        + Channels: 1
        + Sampling rate: 1 hour
        + Missing values: 1.09%
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/electricity/electricity.txt.gz?raw=true'

    similarity_options = None

    default_similarity_score = None
    default_temporal_aggregation = 'sum'
    default_spatial_aggregation = 'sum'
    default_freq = '1H'
    start_date = '01-01-2012 00:00'
    bool_multinode = True

    @property
    def raw_file_names(self):
        return ['electricity.txt.gz']


# at timestep splitter
class ETTSplitter(Splitter):
    def __init__(self, dataset_name, seq_len, horizon):
        self.seq_len = seq_len
        self.horizon = horizon
        self.dataset_name = dataset_name
        super(ETTSplitter, self).__init__()

    def fit(self, dataset: _LTSFDataset) -> dict:
        
        idx = np.arange(len(dataset))
        # len_dataset = len(dataset)
        # test_len = int(0.2 * len(idx))
        # train_len = int(0.7 * len(idx))
        
        # val_len = len_dataset - train_len - test_len
        # test_start = len(idx) - test_len
        # val_end = train_len + val_len
        # self.set_indices(idx[:train_len - self.seq_len - self.horizon],
        #                  idx[train_len:val_end - self.seq_len - self.horizon],
        #                  idx[val_end:])

        # return self.indices
    

        # #TODO: remove this
        if self.dataset_name == 'etth1' or self.dataset_name == 'etth2':
            self.set_indices(idx[:(12 * 30 * 24) - self.seq_len - self.horizon +1 ],
                            idx[(12 * 30 * 24)- self.seq_len  : (12 * 30 * 24 + 4 * 30 * 24) - self.seq_len - self.horizon +1 ],
                            idx[(12 * 30 * 24 + 4 * 30 * 24- self.seq_len ) : 12 * 30 * 24 + 8 * 30 * 24 - self.seq_len - self.horizon+1 ])
        elif self.dataset_name == 'ettm1' or self.dataset_name == 'ettm2':
            self.set_indices(idx[:(12 * 30 * 96) - self.seq_len - self.horizon +1 ],
                            idx[(12 * 30 * 96)- self.seq_len  : (12 * 30 * 96 + 4 * 30 * 96) - self.seq_len - self.horizon +1 ],
                            idx[(12 * 30 * 96 + 4 * 30 * 96)- self.seq_len  : 12 * 30 * 96 + 8 * 30 * 96 - self.seq_len - self.horizon+1 ])
        
        
        return self.indices
    
