import datetime
import numpy as np
import pandas as pd
import io

from typing import Tuple, List
from tqdm import tqdm
from datetime import timedelta

THRESHOLD_MINUTES = 500
TOTAL_FOR_LOOPS = 18
MISSING_DATA_THRESHOLD = 10.0 

def add_schedule_detail(historical_information: pd.DataFrame) -> pd.DataFrame:
    """
    Add schedule detail to the historical information dataset.

    Parameters:
    - historical_information: DataFrame representing the historical information dataset.

    Returns:
    - DataFrame with added schedule detail.
    """
    for i in tqdm(range(len(historical_information))):
        df_schedule_detail = pd.read_csv(io.StringIO(historical_information.iloc[i]['5.schedule_detail']), sep=',', dtype=str)
        df_schedule_detail = df_schedule_detail.drop(df_schedule_detail.columns[0], axis=1)
        historical_information.at[i, '5.schedule_detail'] = df_schedule_detail
    return historical_information

def set_starting_terminating_times(schedule: pd.DataFrame) -> pd.DataFrame:
    """ 
    Set the starting and terminating times for a given schedule.
    
    Parameters:
    - schedule: DataFrame representing the schedule.

    Returns:
    - DataFrame with modified starting and terminating times.
    """
    schedule.iloc[0]['actual_ta'] = 'starting'
    schedule.iloc[0]['gbtt_pta'] = 'starting'
    schedule.iloc[-1]['actual_td'] = 'terminating'
    schedule.iloc[-1]['gbtt_ptd'] = 'terminating'
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

def get_time_difference(time1: str, time2: str, format='%H%M') -> float():
    '''
    Calculate the time difference between two given times.

    Parameters:
    - time1: First time.
    - time2: Second time.

    Returns:
    - Time difference between the two given times in minutes
    '''
    startDateTime = datetime.datetime.strptime(time1, format)
    endDateTime = datetime.datetime.strptime(time2, format)
    diff_minutes = (endDateTime - startDateTime).total_seconds() / 60.0
    if np.abs(diff_minutes) > THRESHOLD_MINUTES:
        endDateTime += timedelta(days=1)
        diff_minutes = (endDateTime - startDateTime).total_seconds() / 60.0
    return diff_minutes

def travel_time(schedule: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the travel time for a given schedule.

    Parameters:
    - schedule: DataFrame representing the schedule.

    Returns:
    - DataFrame with added actual and predicted travel time.
    '''

    travel_times = [0] * len(schedule)
    travel_times_predicted = [0] * len(schedule)

    for j in range(len(schedule) - 1):
        # for loop runs until the penultimate row to avoid index out of bounds error
        # get the actual and public arrival and departure time of the current and next station
        actual_td, actual_ta = schedule.iloc[j]['actual_td'], schedule.iloc[j + 1]['actual_ta']
        gbtt_ptd, gbtt_pta = schedule.iloc[j]['gbtt_ptd'], schedule.iloc[j + 1]['gbtt_pta']

        # if the actual and public arrival and departure time of the current and next station are not null then calculate the travel time
        if isinstance(actual_ta, str) and isinstance(actual_td, str):
            # the actual travel time to get to the current station starting from the second station
            travel_times[j + 1] = get_time_difference(actual_td, actual_ta)
        
        if isinstance(gbtt_pta, str) and isinstance(gbtt_ptd, str):
            # the predicted travel time to get to the current station starting from the second station
            travel_times_predicted[j + 1] = get_time_difference(gbtt_ptd, gbtt_pta)

    # add actual and predicted travel time a new feature to the schedule dataframe
    schedule['travel_time'] = travel_times
    schedule['travel_time_predicted'] = travel_times_predicted
    return schedule

def dwell_time(schedule: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the dwell time for a given schedule.

    Parameters:
    - schedule: DataFrame representing the schedule.

    Returns:
    - DataFrame with added actual and predicted dwell time.
    '''

    dwell_times = [0] * len(schedule)
    dwell_times_predicted = [0] * len(schedule)

    for j in range(len(schedule) - 2):
        # for loop runs from the second row to the penultimate row as the first and last row will be the starting terminating station and not have a dwell time
        actual_ta, actual_td = schedule.iloc[j + 1]['actual_ta'], schedule.iloc[j + 1]['actual_td']
        gbtt_pta, gbtt_ptd = schedule.iloc[j + 1]['gbtt_pta'], schedule.iloc[j + 1]['gbtt_ptd']
        
        # if the actual and public arrival and departure time of the current and next station are not null then calculate the dwell time
        if isinstance(actual_ta, str) and isinstance(actual_td, str):
            # the actual dwell time of the current station next station starting from the second station
            dwell_times[j + 1] = get_time_difference(actual_ta, actual_td)

        if isinstance(gbtt_pta, str) and isinstance(gbtt_ptd, str):
            # the predicted dwell time of the current station next station starting from the second station
            dwell_times_predicted[j + 1] = get_time_difference(gbtt_pta, gbtt_ptd)

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
    
    for i, row in tqdm(enumerate(historical_information.iterrows()), total=len(historical_information), desc=f'Extracting all dwell times'):
        schedule_detail = row[1]['5.schedule_detail']
        
        for _, detail in schedule_detail.iterrows():
            station = detail['location']
            dwell_time = detail['dwell_time']
            dwell_time_predicted = detail['dwell_time_predicted']

            all_stations.append(station)
            dwell_time_all_stations.append(dwell_time)
            dwell_time_predicted_all_stations.append(dwell_time_predicted)

            if (np.absolute(dwell_time) > THRESHOLD_MINUTES) or (np.absolute(dwell_time_predicted) > THRESHOLD_MINUTES):
                extreme_value_index.append(i)
    
    dwell_time_stations = pd.DataFrame({
        '1.station': all_stations,
        '2.dwell_time': dwell_time_all_stations,
        '3.dwell_time_predicted': dwell_time_predicted_all_stations
    })
    
    return dwell_time_stations, extreme_value_index

def dwell_time_summary_statistice(dwell_time_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate unique station metrics such as total and average dwell times.
    
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

    return aggregated

def process_historical_data(historical_information: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    '''
    Process the historical information dataset.

    Parameters:
    - historical_information: DataFrame representing the historical information dataset.

    Returns:
    - Processed historical information dataset.
    '''
    # Set arrival/departure time for starting/terminating stations
    print('Setting arrival/departure time for starting/terminating stations...')
    historical_information.loc[:,'5.schedule_detail'] = historical_information['5.schedule_detail'].apply(set_starting_terminating_times)

    # Calculate missing data percentages
    print('Calculating missing data percentages...')
    historical_information.loc[:,'percentage_null'] = historical_information['5.schedule_detail'].apply(calculate_missing_percentage)

    # Drop rows with more than 10% missing data
    print('Dropping rows with more than 10% missing data...')
    historical_information = historical_information[historical_information['percentage_null'] <= MISSING_DATA_THRESHOLD].reset_index(drop=True)

    # Extract trips to find unique ones
    print('Extracting trips to find unique ones...')
    historical_information['locations'] = historical_information['5.schedule_detail'].apply(lambda x: x['location'].to_string())
    duplicate_list = historical_information.duplicated(['2.origin', '3.destination', '4.stops', 'locations'], keep='first').tolist()
    historical_information['duplicate_list'] = duplicate_list

    # Get unique trips
    print('Getting unique trips...')
    unique_trips = historical_information[~historical_information.duplicate_list]
    unique_trips = unique_trips.drop(columns=['1.date', 'percentage_null', 'duplicate_list']).sort_values(by='2.origin').reset_index(drop=True)

    # Calculate time-related metrics
    for index, row in tqdm(historical_information.iterrows(), desc='Calculate time taken to travel and dwell time of all trips'):
        schedule = row['5.schedule_detail']

        # calculate the actual and predicted travel time for the current schedule
        schedule = travel_time(schedule)

        # calculate the actual and predicted dwell time for the current schedule
        schedule = dwell_time(schedule)

        # Update the original dataframe with the processed schedule
        historical_information.at[index, '5.schedule_detail'] = schedule
    
    # Extract dwell times for each station
    print('Extracting dwell times for each station...')
    dwell_time_stations, extreme_value_index = dwell_time_extract(historical_information)
    
    # Calculate summary statistics for dwell times
    print('Calculating summary statistics for dwell times...')
    aggregated = dwell_time_summary_statistice(dwell_time_stations)

    return historical_information, unique_trips

if __name__ == '__main__':

    DATA_PATHS = ["Data/hist_info_DID_PAD_2016.csv"]
    OUTPUT_FILENAME = "Data/feature_engineered.csv"  # Adjust this accordingly, or include the dynamic filename generation logic.

    historical_information = pd.read_csv(DATA_PATHS[0])
    historical_infromation_test = historical_information.head(1000)
    historical_infromation_test = add_schedule_detail(historical_infromation_test)
    # process historical information to add actual and predicted travel time and dwell time and get unique trips
    historical_information_refactored, unique_trips = process_historical_data(historical_infromation_test)
    historical_information_refactored.to_csv(OUTPUT_FILENAME)

