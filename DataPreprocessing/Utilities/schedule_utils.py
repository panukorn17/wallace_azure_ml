import io
import pandas as pd
import numpy as np
import datetime

from typing import Tuple, List
from tqdm import tqdm

# Import from time_utils.py
from Utilities.datetime_utils import travel_time
from Utilities.datetime_utils import dwell_time
from Utilities.datetime_utils import merge_travel_times
from Utilities.datetime_utils import merge_dwell_times
from Utilities.datetime_utils import get_unique_dates
from Utilities.datetime_utils import extract_origin_departure_time

# Import from condition_utils.py
from Utilities.condition_utils import process_condition_1_aa
from Utilities.condition_utils import process_condition_2_aa
from Utilities.condition_utils import process_condition_3_aa
from Utilities.condition_utils import process_condition_1_ad
from Utilities.condition_utils import process_condition_2_ad
from Utilities.condition_utils import process_condition_3_ad

def calculate_metrics(schedule: pd.DataFrame, STATION_OFFSET_DWELL_TIME: int, FORMAT: str) -> pd.DataFrame:
    """
    Calculate travel and dwell times.
    
    Parameters:
    - row: DataFrame representing the schedule.
    
    Returns:
    - DataFrame with added actual and predicted travel and dwell times.
    """
    schedule = travel_time(schedule, FORMAT)
    schedule = dwell_time(schedule, STATION_OFFSET_DWELL_TIME, FORMAT)
    return schedule

def add_schedule_detail(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Add schedule detail to a given schedule.

    Parameters:
    - schedule: DataFrame representing the schedule.

    Returns:
    - DataFrame of schedule schdule.
    """
    schedule = pd.read_csv(io.StringIO(schedule), sep=',', dtype=str, engine='python')
    schedule = schedule.drop(schedule.columns[0], axis=1)
    return schedule

def calculate_missing_percentage(schedule: pd.DataFrame) -> float():
    '''
    Calculate the percentage of missing data in a given schedule.

    Parameters:
    - schedule: DataFrame representing the schedule.

    Returns:
    - Percentage of missing data in the schedule.
    '''
    schedule = schedule.replace(r'\s+', np.nan, regex=True).replace('', np.nan)
    return schedule.isnull().sum().sum() / np.prod(schedule.shape) * 100

def get_total_null(historical_information: pd.DataFrame) -> pd.DataFrame:
    '''
    Compute total null value.

    Parameters:
    - historical_information: DataFrame representing the historical information dataset.

    Returns:
    - Total null value.
    '''
    total_null = historical_information['5.schedule_detail'].apply(
        lambda x: x.loc[:, 'gbtt_ptd':'actual_ta'].isnull().sum().sum()
    ).sum()
    return total_null

def extract_schedule_OD_travel(historical_information: pd.DataFrame, STATION_OFFSET_TRAVEL_TIME: int) -> Tuple[List[str], List[str], List[float], List[float]]:
    """
    Extract details from schedule.
    
    Parameters:
    - historical_information: DataFrame containing the historical data.

    Returns:
    - List of origins.
    - List of destinations.
    - List of travel times.
    - List of predicted travel times.
    """
    origins, destinations, travel_times, predicted_travel_times = [], [], [], []
    
    for _, row in historical_information.iterrows():
        schedule_detail = row['5.schedule_detail']
        
        # Extract details of origin and destination travel time and predicted travel time in minutes
        for i in range(len(schedule_detail) - STATION_OFFSET_TRAVEL_TIME):
            origins.append(schedule_detail.iloc[i]['location'])
            destinations.append(schedule_detail.iloc[i + STATION_OFFSET_TRAVEL_TIME]['location'])
            travel_times.append(schedule_detail.iloc[i + STATION_OFFSET_TRAVEL_TIME]['travel_time'])
            predicted_travel_times.append(schedule_detail.iloc[i + STATION_OFFSET_TRAVEL_TIME]['travel_time_predicted'])
            
    return origins, destinations, travel_times, predicted_travel_times


def create_OD_pairs_dataframe(origins: List[str], destinations: List[str], travel_times: List[float], predicted_travel_times: List[float]) -> pd.DataFrame:
    """
    Create the OD pairs DataFrame.
    
    Parameters:
    - origins: List of origins.
    - destinations: List of destinations.
    - travel_times: List of travel times.
    - predicted_travel_times: List of predicted travel times.

    Returns:
    - DataFrame containing OD pairs.
    """
    return pd.DataFrame({
        '1.origin': origins,
        '2.destination': destinations,
        '3.time_travelled': travel_times,
        '4.predicted_time_travelled': predicted_travel_times
    })


def drop_nan_pairs(historical_information: pd.DataFrame, OD_pairs_unique: pd.DataFrame, NEXT_STATION_OFFSET: int)-> pd.DataFrame:
    """
    Drop rows with nan values.

    Parameters:
    - historical_information: DataFrame containing the historical data.
    - OD_pairs_unique: DataFrame containing OD pairs.

    Returns:
    - DataFrame with dropped rows.
    """
    # Extract list of lists containing the origin and destinations that have nan average travel times
    OD_pairs_unique_nan = OD_pairs_unique[np.isnan(OD_pairs_unique['average_travel_time'])][['1.origin', '2.destination']].values.tolist()
    # Convert each inner list into a tuple. The result is a list of tuples, where each tuple represents an (origin, destination) pair.
    OD_pairs_unique_nan = [tuple(i) for i in OD_pairs_unique_nan]
    # Check for each row if it contains any OD nan pair
    to_drop = historical_information['5.schedule_detail'].apply(
        lambda x: any((row['location'], x.loc[idx + NEXT_STATION_OFFSET, 'location']) in OD_pairs_unique_nan for idx, row in x.iterrows() if idx + NEXT_STATION_OFFSET < len(x))
    )
    return historical_information[~to_drop].reset_index(drop=True)

def drop_all_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where all data is null.

    Parameters:
    - df: DataFrame.

    Returns:
    - DataFrame with dropped rows.
    """
    return df.dropna(axis=0, how='all')

def extract_missing_aa_data(historical_information: pd.DataFrame, NEXT_STATION_OFFSET: int) -> pd.DataFrame:
    '''
    Extract data with missing actual arrival times and their indexes in the historical information dataframe and schedule dataframe.

    Parameters:
    - historical_information: DataFrame representing the historical information dataset.

    Returns:
    - DataFrame containing missing data.
    '''
    rows_list = []
    for idx, row in historical_information.iterrows():
        x = row['5.schedule_detail']
        for j in x.index[:-1]:
            if pd.isna(x.loc[j + NEXT_STATION_OFFSET, 'actual_ta']):
                data = {
                    '1.origin': x.loc[j, 'location'],
                    '2.destination': x.loc[j + NEXT_STATION_OFFSET, 'location'],
                    '3.actual_departure_prev_station': x.loc[j, 'actual_td'],
                    '4.actual_arrival_curr_station': x.loc[j + NEXT_STATION_OFFSET, 'actual_ta'],
                    '5.actual_departure_curr_station': x.loc[j + NEXT_STATION_OFFSET, 'actual_td'],
                    '6.h_i_index': idx,
                    '7.s_d_index': j + NEXT_STATION_OFFSET,
                    'average_actual_travel_time': 0,
                    'average_predicted_travel_time': 0,
                    'average_actual_dwell_time': 0,
                    'average_predicted_dwell_time': 0
                }
                rows_list.append(data)

    return pd.DataFrame(rows_list)

def extract_missing_ad_data(historical_information: pd.DataFrame, NEXT_STATION_OFFSET: int) -> pd.DataFrame:
    '''
    Extract data with missing actual departure times and their indexes in the historical information dataframe and schedule dataframe.

    Parameters:
    - historical_information: DataFrame representing the historical information dataset.

    Returns:
    - DataFrame containing missing data.
    '''
    rows_list = []
    for idx, row in historical_information.iterrows():
        x = row['5.schedule_detail']
        for j in x.index[:-1]:
            if pd.isna(x.loc[j, 'actual_td']):
                data = {
                    '1.origin': x.loc[j, 'location'],
                    '2.destination': x.loc[j + NEXT_STATION_OFFSET, 'location'],
                    '3.actual_arrival_curr_station': x.loc[j, 'actual_ta'],
                    '4.actual_departure_curr_station': x.loc[j, 'actual_td'],
                    '5.actual_arrival_next_station': x.loc[j + NEXT_STATION_OFFSET, 'actual_ta'],
                    '6.h_i_index': idx,
                    '7.s_d_index': j,
                    'average_actual_travel_time': 0,
                    'average_predicted_travel_time': 0,
                    'average_actual_dwell_time': 0,
                    'average_predicted_dwell_time': 0
                }
                rows_list.append(data)

    return pd.DataFrame(rows_list)

def impute_missing_actual_arrival(aa_nan: pd.DataFrame, historical_information: pd.DataFrame, format: str):
    '''
    Impute missing actual arrival time into the historical_information dataframe.

    Parameters:
    - aa_nan: DataFrame containing missing data.
    - historical_information: DataFrame representing the historical information dataset.
    - format: Format of time strings.
    '''
    # Initialize column with default values
    aa_nan['actual_arrival_time_1'] = [0] * len(aa_nan)

    for i in range(0, len(aa_nan)):
        row = aa_nan.iloc[i]

        prev_dep_str = row['3.actual_departure_prev_station']
        curr_dep_str = row['5.actual_departure_curr_station']

        # Condition 1: both previous and current stations have data and current isn't terminating
        if pd.notna(prev_dep_str) and pd.notna(curr_dep_str) and curr_dep_str != 'terminating':
            arrival_time = process_condition_1_aa(row, format)

        # Condition 2: only previous station has data or current is terminating
        elif pd.notna(prev_dep_str) and (pd.isna(curr_dep_str) or curr_dep_str == 'terminating'):
            arrival_time = process_condition_2_aa(row, format)

        # Condition 3: only current station has data and it's not terminating
        elif pd.isna(prev_dep_str) and pd.notna(curr_dep_str) and curr_dep_str != 'terminating':
            arrival_time = process_condition_3_aa(row, format)
        else:
            arrival_time = None

        # Set the computed values
        aa_nan.at[i, 'actual_arrival_time_1'] = arrival_time
        aa_nan.at[i, '4.actual_arrival_curr_station'] = arrival_time
        h_i_index = aa_nan.at[i, '6.h_i_index']
        s_d_index = aa_nan.at[i, '7.s_d_index']
        historical_information.at[h_i_index, '5.schedule_detail'].at[s_d_index, 'actual_ta'] = arrival_time

def impute_missing_actual_departure(ad_nan: pd.DataFrame, historical_information: pd.DataFrame, format: str):
    '''
    Impute missing actual departure time into the historical_information dataframe.

    Parameters:
    - ad_nan: DataFrame containing missing data.
    - historical_information: DataFrame representing the historical information dataset.

    Returns:
    - DataFrame with imputed missing data.
    '''
    # Initialize column with default values
    ad_nan['actual_departure_time_1'] = [0] * len(ad_nan)

    for i in range(0,len(ad_nan)):
        row = ad_nan.iloc[i]

        curr_arr_str = row['3.actual_arrival_curr_station']
        next_arr_str = row['5.actual_arrival_next_station']
        
        # calculate the departure time from the current station based on the arrival time of the of the next station adn the average travel time

        # Condition 1: when both previous and current stations have data and the next station isn't starting
        if pd.notna(curr_arr_str) and pd.notna(next_arr_str) and curr_arr_str != 'starting':
            departure_time = process_condition_1_ad(row, format)

        # Condition 2: when next station arrival has data and current station arrival has no data or is starting
        elif pd.notna(next_arr_str) and ((pd.isna(curr_arr_str) or curr_arr_str == 'starting')):
            departure_time = process_condition_2_ad(row, format)

        # Condition 3: when the current station has data and is starting and next station arrival has no data
        elif pd.notna(curr_arr_str) and curr_arr_str != 'starting' and pd.isna(next_arr_str):
            departure_time = process_condition_3_ad(row, format)

        else:
            departure_time = None

        # Set the computed values
        ad_nan.at[i, 'actual_departure_time_1'] = departure_time
        ad_nan.at[i, '4.actual_departure_curr_station'] = departure_time
        h_i_index = ad_nan.at[i, '6.h_i_index']
        s_d_index = ad_nan.at[i, '7.s_d_index']
        historical_information.at[h_i_index, '5.schedule_detail'].at[s_d_index, 'actual_td'] = departure_time

def impute_missing_data(historical_information: pd.DataFrame, od_pairs_unique: pd.DataFrame, station_dwell_time_unique: pd.DataFrame, NEXT_STATION_OFFSET: int, FORMAT: str):
    '''
    Impute missing data into the historical_information dataframe.

    Parameters:
    - historical_information: DataFrame representing the historical information dataset.
    - od_pairs_unique: DataFrame containing OD pairs.
    - station_dwell_time_unique: DataFrame containing station dwell time data.

    Returns:
    - DataFrame with imputed missing data.
    '''    
    # Compute missing data stats
    total_null = get_total_null(historical_information)
    print('Total null values =', total_null)

    # Initialize total null new
    total_null_new = 0

    # while the total null reduces, keep imputing missing data
    while abs(total_null - total_null_new) > 0:
        # Update total null
        total_null_new = total_null
        # Extract missing actual arrival times
        print('Extracting missing actual arrival data...')
        aa_nan = extract_missing_aa_data(historical_information, NEXT_STATION_OFFSET)
        aa_nan = drop_all_null_rows(aa_nan)
        # check null:
        if not(aa_nan.empty):
            aa_nan = merge_travel_times(aa_nan, od_pairs_unique)
            aa_nan = merge_dwell_times(aa_nan, station_dwell_time_unique)
            # Impute missing actual arrival time into the historical_information dataframe and update the actual arrival missing dataframe
            impute_missing_actual_arrival(aa_nan, historical_information, FORMAT)
        # Extract missing actual departure times
        print('Extracting missing actual departure data...')
        ad_nan = extract_missing_ad_data(historical_information, NEXT_STATION_OFFSET)
        ad_nan = drop_all_null_rows(ad_nan)
        # check null:
        if not(ad_nan.empty):
            ad_nan = merge_travel_times(ad_nan, od_pairs_unique)
            ad_nan = merge_dwell_times(ad_nan, station_dwell_time_unique)
            # Impute missing actual departure time into the historical_information dataframe and update the actual departure missing dataframe
            impute_missing_actual_departure(ad_nan, historical_information, FORMAT)
        # Compute missing data stats
        total_null = get_total_null(historical_information)
        print('Total null values =', total_null)

def compute_departure_order_for_date(historical_data: pd.DataFrame, date: datetime)-> List[int]:
    """
    Computes the departure order for a given date.

    Parameters:
    - historical_data: DataFrame representing the historical data.
    - date: Date.

    Returns:
    - List of departure orders for a given date.
    """
    day_schedule = historical_data[historical_data['1.date'] == date]
    return list(range(1, len(day_schedule) + 1))

def compute_departure_order(historical_data: pd.DataFrame)-> List[int]:
    """
    Computes the departure order for each record in historical data.

    Parameters:
    - historical_data: DataFrame representing the historical data.

    Returns:
    - List of departure orders.
    """
    unique_dates = get_unique_dates(historical_data)
    departure_orders = []

    for date in tqdm(unique_dates):
        departure_orders.extend(compute_departure_order_for_date(historical_data, date))

    return departure_orders

def assign_departure_order(historical_data: pd.DataFrame)-> pd.DataFrame:
    """
    Assigns departure order to each record in historical data.

    Parameters:
    - historical_data: DataFrame representing the historical data.

    Returns:
    - DataFrame with assigned departure order.
    """
    # Extract origin departure times
    historical_data['origin_departure_time'] = historical_data['5.schedule_detail'].apply(extract_origin_departure_time)

    # Sort by date and origin departure time
    historical_data.sort_values(by=['1.date', 'origin_departure_time'], inplace=True)
    historical_data.reset_index(inplace=True, drop=True)

    # Compute and assign departure order
    historical_data['departure_order'] = compute_departure_order(historical_data)

    return historical_data