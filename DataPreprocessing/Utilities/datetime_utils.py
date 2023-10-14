import datetime
import pandas as pd
import numpy as np
from datetime import timedelta

from typing import Tuple, List
from tqdm import tqdm

THRESHOLD_MINUTES = 500
MISSING_DATA_THRESHOLD_INITIAL = 10.0 
MISSING_DATA_THRESHOLD_FINAL = 0
STATION_OFFSET_TRAVEL_TIME = 1
NEXT_STATION_OFFSET = 1
FORMAT = '%H%M'

def is_valid_string(value):
    return value is not None and isinstance(value, str) and value != ""

def set_starting_terminating_times(schedule: pd.DataFrame) -> pd.DataFrame:
    """ 
    Set the starting and terminating times for a given schedule.
    
    Parameters:
    - schedule: DataFrame representing the schedule.

    Returns:
    - DataFrame with modified starting and terminating times.
    """
    schedule.at[0,'actual_ta'] = 'starting'
    schedule.at[0,'gbtt_pta'] = 'starting'
    schedule.iloc[-1, schedule.columns.get_loc('actual_td')] = 'terminating'
    schedule.iloc[-1, schedule.columns.get_loc('gbtt_ptd')] = 'terminating'
    return schedule

def get_time_difference(time1: str, time2: str, FORMAT: str) -> float():
    '''
    Calculate the time difference between two given times.

    Parameters:
    - time1: First time.
    - time2: Second time.

    Returns:
    - Time difference between the two given times in minutes
    '''
    startDateTime = datetime.datetime.strptime(time1, FORMAT)
    endDateTime = datetime.datetime.strptime(time2, FORMAT)
    diff_minutes = (endDateTime - startDateTime).total_seconds() / 60.0
    if np.abs(diff_minutes) > THRESHOLD_MINUTES:
        endDateTime += timedelta(days=1)
        diff_minutes = (endDateTime - startDateTime).total_seconds() / 60.0
    return diff_minutes

def travel_time(schedule: pd.DataFrame, FORMAT: str) -> pd.DataFrame:
    '''
    Calculate the travel time for a given schedule.

    Parameters:
    - schedule: DataFrame representing the schedule.

    Returns:
    - DataFrame with added actual and predicted travel time.
    '''

    travel_times = [0] * len(schedule)
    travel_times_predicted = [0] * len(schedule)

    for j in range(len(schedule) - STATION_OFFSET_TRAVEL_TIME):
        # for loop runs until the penultimate row to avoid index out of bounds error
        # get the actual and public arrival and departure time of the current and next station
        actual_td, actual_ta = schedule.iloc[j]['actual_td'], schedule.iloc[j + NEXT_STATION_OFFSET]['actual_ta']
        gbtt_ptd, gbtt_pta = schedule.iloc[j]['gbtt_ptd'], schedule.iloc[j + NEXT_STATION_OFFSET]['gbtt_pta']

        # if the actual and public arrival and departure time of the current and next station are not null then calculate the travel time
        if is_valid_string(actual_ta) and is_valid_string(actual_td):
            # the actual travel time to get to the current station starting from the second station
            travel_times[j + NEXT_STATION_OFFSET] = get_time_difference(actual_td, actual_ta, FORMAT)
        
        if is_valid_string(gbtt_pta) and is_valid_string(gbtt_ptd):
            # the predicted travel time to get to the current station starting from the second station
            travel_times_predicted[j + NEXT_STATION_OFFSET] = get_time_difference(gbtt_ptd, gbtt_pta, FORMAT)

    # add actual and predicted travel time a new feature to the schedule dataframe
    schedule['travel_time'] = travel_times
    schedule['travel_time_predicted'] = travel_times_predicted
    return schedule

def dwell_time(schedule: pd.DataFrame, STATION_OFFSET_DWELL_TIME: int, FORMAT: str) -> pd.DataFrame:
    '''
    Calculate the dwell time for a given schedule.

    Parameters:
    - schedule: DataFrame representing the schedule.

    Returns:
    - DataFrame with added actual and predicted dwell time.
    '''

    dwell_times = [0] * len(schedule)
    dwell_times_predicted = [0] * len(schedule)

    for j in range(len(schedule) - STATION_OFFSET_DWELL_TIME):
        # for loop runs from the second row to the penultimate row as the first and last row will be the starting terminating station and not have a dwell time
        actual_ta, actual_td = schedule.iloc[j + NEXT_STATION_OFFSET]['actual_ta'], schedule.iloc[j + NEXT_STATION_OFFSET]['actual_td']
        gbtt_pta, gbtt_ptd = schedule.iloc[j + NEXT_STATION_OFFSET]['gbtt_pta'], schedule.iloc[j + NEXT_STATION_OFFSET]['gbtt_ptd']
        
        # if the actual and public arrival and departure time of the current and next station are not null then calculate the dwell time
        if is_valid_string(actual_ta) and is_valid_string(actual_td):
            # the actual dwell time of the current station next station starting from the second station
            dwell_times[j + NEXT_STATION_OFFSET] = get_time_difference(actual_ta, actual_td, FORMAT)

        if is_valid_string(gbtt_pta) and is_valid_string(gbtt_ptd):
            # the predicted dwell time of the current station next station starting from the second station
            dwell_times_predicted[j + NEXT_STATION_OFFSET] = get_time_difference(gbtt_pta, gbtt_ptd, FORMAT)

    # add actual and predicted dwell time a new feature to the schedule dataframe
    schedule['dwell_time'] = dwell_times
    schedule['dwell_time_predicted'] = dwell_times_predicted
    return schedule

def dwell_time_extract(historical_information: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """
    Extracts dwell times for each station from the historical information.
    
    Parameters:
    - historical_information: DataFrame containing the historical data.
    - total_for_loops: Total number of loops for progress bar.
    
    Returns:
    - DataFrame containing station, dwell time, and predicted dwell time.
    - List of indices of extreme values.
    """
    
    all_stations = []
    dwell_time_all_stations = []
    dwell_time_predicted_all_stations = []
    extreme_value_index = []
    
    # Iterate through the DataFrame index
    for i in tqdm(historical_information.index, desc=f'Extracting all dwell times'):
        schedule_detail = historical_information.at[i, '5.schedule_detail']
        
        # Extract the necessary columns from schedule_detail
        stations = schedule_detail['location'].tolist()
        dwell_times = schedule_detail['dwell_time'].tolist()
        dwell_times_predicted = schedule_detail['dwell_time_predicted'].tolist()

        # Append to results lists
        all_stations.extend(stations)
        dwell_time_all_stations.extend(dwell_times)
        dwell_time_predicted_all_stations.extend(dwell_times_predicted)
        
        # Check threshold conditions for each dwell time and predicted dwell time
        for dwell_time, dwell_time_predicted in zip(dwell_times, dwell_times_predicted):
            if (np.absolute(dwell_time) > THRESHOLD_MINUTES) or (np.absolute(dwell_time_predicted) > THRESHOLD_MINUTES):
                extreme_value_index.append(i)
    
    dwell_time_stations = pd.DataFrame({
        '1.station': all_stations,
        '2.dwell_time': dwell_time_all_stations,
        '3.dwell_time_predicted': dwell_time_predicted_all_stations
    })
    
    return dwell_time_stations, extreme_value_index

def dwell_time_summary_statistics(dwell_time_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineer unique station metrics such as total and average dwell times.
    
    Parameters:
    - dwell_time_stations: DataFrame with station dwell time data.

    Returns:
    - DataFrame with unique station metrics.
    """
    aggregated = dwell_time_stations.groupby('1.station').agg(
        total_dwell_time=pd.NamedAgg(column='2.dwell_time', aggfunc='sum'),
        total_dwell_time_predicted=pd.NamedAgg(column='3.dwell_time_predicted', aggfunc='sum'),
        average_dwell_time=pd.NamedAgg(column='2.dwell_time', aggfunc='mean'),
        average_dwell_time_predicted=pd.NamedAgg(column='3.dwell_time_predicted', aggfunc='mean')
    ).reset_index()

    # Remove rows with all zero values
    aggregated = aggregated.loc[~(aggregated == 0).all(axis=1)]

    return aggregated

def travel_time_summary_statistics(OD_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate travel times summmary statistics.
    
    Parameters:
    - OD_pairs: DataFrame containing OD pairs.

    Returns:
    - DataFrame containing travel times summary statistics.
    """
    # Group by origin and destination
    group = OD_pairs.groupby(['1.origin', '2.destination'])
    
    # Calculate aggregate values
    agg_df = group.agg(total_travel_time=('3.time_travelled', 'sum'),
                       average_travel_time=('3.time_travelled', 'mean'),
                       total_travel_time_predicted=('4.predicted_time_travelled', 'sum'),
                       average_travel_time_predicted=('4.predicted_time_travelled', 'mean')).reset_index()
    
    return agg_df

def merge_travel_times(actual_nan: pd.DataFrame, OD_pairs_unique: pd.DataFrame) -> pd.DataFrame:
    """
    Merge average actual and predicted travel times.
    
    Parameters:
    - actual_nan: DataFrame containing missing actual data.
    - OD_pairs_unique: DataFrame containing OD pairs.
    
    Returns:
    - DataFrame with merged average actual and predicted travel times.
    """
    # Similar to SQL left join, match the origin and destination columns in actual_nan and OD_pairs_unique and merge the two dataframes
    # to get the average actual and predicted travel times for the missing data
    merged = pd.merge(
        actual_nan, 
        OD_pairs_unique[['1.origin', '2.destination', 'average_travel_time', 'average_travel_time_predicted']], 
        on=['1.origin', '2.destination'], 
        how='left'
    )
    actual_nan['average_actual_travel_time'] = merged['average_travel_time']
    actual_nan['average_predicted_travel_time'] = merged['average_travel_time_predicted']
    return actual_nan

def merge_dwell_times(actual_nan: pd.DataFrame, station_dwell_time_unique: pd.DataFrame) -> pd.DataFrame:
    """
    Merge average dwell and predicted dwell times.
    
    Parameters:
    - actual_nan: DataFrame containing missing data.
    - station_dwell_time_unique: DataFrame containing station dwell time data.
    
    Returns:
    - DataFrame with merged average dwell and predicted dwell times.
    """
    # Similar to SQL left join, match the destination column in actual_nan and station_dwell_time_unique and merge the two dataframes
    # to get the average actual and predicted dwell times for the missing data
    merged = pd.merge(
        actual_nan,
        station_dwell_time_unique[['1.station', 'average_dwell_time', 'average_dwell_time_predicted']],
        left_on='2.destination',
        right_on='1.station',
        how='left'
    )
    actual_nan['average_actual_dwell_time'] = merged['average_dwell_time']
    actual_nan['average_predicted_dwell_time'] = merged['average_dwell_time_predicted']
    return actual_nan

def str_to_datetime(date_series: pd.Series, format: str)-> pd.Series:
    """
    Convert Series of date strings to datetime.
    
    Parameters:
    - date_series: Series of date strings.
    - format: Format of date strings.
    
    Returns:
    - Series of datetime.
    """
    return pd.to_datetime(date_series, format=format, errors='coerce')

def compute_arrival_from_prev(prev_station_dep: datetime, avg_travel_time: float, format: str)-> datetime:
    """
    Compute arrival time based on previous station's departure.
    
    Parameters:
    - prev_station_dep: Previous station's departure time.
    - avg_travel_time: Average travel time.
    - format: Format of time strings.

    Returns:
    - Arrival time.
    """
    if prev_station_dep:
        return (prev_station_dep + timedelta(minutes=avg_travel_time)).strftime(format)
    return None

def compute_arrival_from_current(curr_station_dep: datetime, avg_dwell_time: float, format: str)-> datetime:
    """
    Compute arrival time based on current station's departure.
    
    Parameters:
    - curr_station_dep: Current station's departure time.
    - avg_dwell_time: Average dwell time.
    - format: Format of time strings.

    Returns:
    - Arrival time.
    """
    if curr_station_dep:
        return (curr_station_dep - timedelta(minutes=avg_dwell_time)).strftime(format)
    return None

def compute_departure_from_next(next_station_arr: datetime, avg_travel_time: float, format: str)-> datetime:
    """
    Compute departure time based on next station's arrival.
    
    Parameters:
    - next_station_arr: Next station's arrival time.
    - avg_travel_time: Average travel time.
    - format: Format of time strings.

    Returns:
    - Departure time.
    """
    if next_station_arr:
        return (next_station_arr - timedelta(minutes=avg_travel_time)).strftime(format)
    return None

def compute_departure_from_current(curr_station_arr: datetime, avg_dwell_time: float, format: str)-> datetime:
    """
    Compute departure time based on current station's arrival.
    
    Parameters:
    - curr_station_arr: Current station's arrival time.
    - avg_dwell_time: Average dwell time.
    - format: Format of time strings.

    Returns:
    - Departure time.
    """
    if curr_station_arr:
        return (curr_station_arr + timedelta(minutes=avg_dwell_time)).strftime(format)
    return None


def extract_origin_departure_time(schedule_detail: pd.DataFrame)-> datetime:
    """
    Extracts the origin departure time from a schedule detail.

    Parameters:
    - schedule_detail: DataFrame representing a schedule detail.

    Returns:
    - Origin departure time.
    """
    return schedule_detail.loc[0, 'actual_td']

def get_unique_dates(historical_data: pd.DataFrame)-> pd.DataFrame:
    """
    Extracts unique dates from the historical data.

    Parameters:
    - historical_data: DataFrame representing the historical data.

    Returns:
    - DataFrame containing unique dates of the historical data.
    """
    dates = historical_data['1.date'].to_frame()
    is_duplicate = dates.duplicated('1.date', keep='first')
    return dates[~is_duplicate].reset_index(drop=True)['1.date']
