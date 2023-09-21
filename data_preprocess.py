import io
import pandas as pd
import os
import sys
from tqdm import tqdm

class DataProcessor:
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.df = self._load_data()
        self.df_imputed = None
        self.station_dwell_avg = None
        self.OD_pairs = None

    def _load_data(self):
        dfs = [pd.read_csv(path) for path in self.data_paths]
        return pd.concat(dfs).reset_index(drop=True)

    def add_schedule_detail(self):
        df_schedule_detail_list = []
        for i in tqdm(range(len(self.df))):
            df_schedule_detail = pd.read_csv(io.StringIO(self.df.iloc[i]['5.schedule_detail']), sep=',', dtype=str)
            df_schedule_detail = df_schedule_detail.drop(df_schedule_detail.columns[0], axis=1)
            df_schedule_detail_list.append(df_schedule_detail)
        self.df.drop("5.schedule_detail", axis=1, inplace=True)
        self.df["5.schedule_detail"] = df_schedule_detail_list

    def process_data(self, feature_engineer, df_next_n_stations, add_delay_mechanism):
        self.df_imputed, self.station_dwell_avg, self.OD_pairs = feature_engineer(self.df)
        self.df = df_next_n_stations(self.df_imputed, self.station_dwell_avg, self.OD_pairs, 1)
        self.df = add_delay_mechanism(self.df)

    def save_to_csv(self, filename):
        self.df.to_csv(filename, index=False)


if __name__ == '__main__':
    # Assuming the notebook is in the same directory as your DataPreprocessing folder
    current_path = os.getcwd()
    dir_path = os.path.join(current_path, "DataPreprocessing")
    sys.path.insert(0, dir_path)
    
    from feature_engineer import feature_engineer
    from df_next_n_stations import df_next_n_stations
    from add_delay_mechanism import add_delay_mechanism
    
    DATA_PATHS = ["Data/hist_info_DID_PAD_2016.csv", "Data/hist_info_PAD_DID_2016.csv"]
    OUTPUT_FILENAME = "Data/data_next_1_station.csv"  # Adjust this accordingly, or include the dynamic filename generation logic.

    processor = DataProcessor(DATA_PATHS)
    processor.add_schedule_detail()
    processor.process_data(feature_engineer, df_next_n_stations, add_delay_mechanism)
    processor.save_to_csv(OUTPUT_FILENAME)