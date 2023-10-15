import io
import pandas as pd
import os
import sys
from tqdm import tqdm

from Utilities.schedule_utils import add_schedule_detail

class DataProcessor:
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.df = self._load_data()
        self.df_imputed = None
        self.station_dwell_avg = None
        self.OD_pairs = None

    def _load_data(self):
        dfs = [pd.read_csv(path) for path in self.data_paths]
        return pd.concat(dfs).reset_index(drop=True).head(1000)

    def add_schedule_df_detail(self):
        print("Adding schedule detail...")
        self.df.loc[:,'5.schedule_detail'] = self.df['5.schedule_detail'].progress_apply(add_schedule_detail)

    def process_data(self, process_historical_data, df_next_n_stations, add_delay_mechanism):
        self.df_imputed, self.station_dwell_avg, self.OD_pairs = process_historical_data(self.df)
        self.df = df_next_n_stations(self.df_imputed, self.station_dwell_avg, self.OD_pairs, 1)
        self.df = add_delay_mechanism(self.df)

    def save_to_csv(self, filename):
        self.df.to_csv(filename, index=False)


if __name__ == '__main__':
    # Assuming the notebook is in the same directory as your DataPreprocessing folder
    current_path = os.getcwd()
    dir_path = os.path.join(current_path, "DataPreprocessing")
    sys.path.insert(0, dir_path)
    
    from feature_engineer_refactored import process_historical_data
    from df_next_n_stations import df_next_n_stations
    from add_delay_mechanism import add_delay_mechanism
    
    DATA_PATHS = ["./Data/hist_info_DID_PAD_2016.csv", "./Data/hist_info_PAD_DID_2016.csv"]
    OUTPUT_FILENAME = "./Data/data_next_1_station.csv"  # Adjust this accordingly, or include the dynamic filename generation logic.

    processor = DataProcessor(DATA_PATHS)
    processor.add_schedule_df_detail()
    processor.process_data(process_historical_data, df_next_n_stations, add_delay_mechanism)
    processor.save_to_csv(OUTPUT_FILENAME)