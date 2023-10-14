import pandas as pd
import numpy as np
import datetime

from Utilities.datetime_utils import str_to_datetime
from Utilities.datetime_utils import compute_arrival_from_prev
from Utilities.datetime_utils import compute_arrival_from_current
from Utilities.datetime_utils import compute_departure_from_next
from Utilities.datetime_utils import compute_departure_from_current

def process_condition_1_aa(row: pd.DataFrame, format: str)-> datetime:
    '''
    Process the arrival time for condition 1. This is when the arrive time calculated from the previous station is greater than the actual departure time of the current station.

    Parameters:
    - row: DataFrame representing a row in the schedule.
    - format: Format of time strings.

    Returns:
    - Arrival time.
    '''
    prev_dep_str = row['3.actual_departure_prev_station']
    curr_dep_str = row['5.actual_departure_curr_station']
    prev_station_dep = str_to_datetime(prev_dep_str, format)
    curr_station_dep = str_to_datetime(curr_dep_str, format)
    avg_travel_time = row['average_predicted_travel_time']
    avg_dwell_time = row['average_predicted_dwell_time']

    arrival_time = compute_arrival_from_prev(prev_station_dep, avg_travel_time, format)
    if arrival_time and arrival_time > curr_dep_str:
        arrival_time = compute_arrival_from_current(curr_station_dep, avg_dwell_time, format)
        if arrival_time and arrival_time < prev_dep_str:
            arrival_time = None
    return arrival_time

def process_condition_1_ad(row: pd.DataFrame, format: str)-> datetime:
    '''
    Process the departure time for condition 1. This is when the departure time calculated from the next station is greater than the actual departure time of the current station.

    Parameters:
    - row: DataFrame representing a row in the schedule.
    - format: Format of time strings.

    Returns:
    - Departure time.
    '''
    curr_arr_str = row['3.actual_arrival_curr_station']
    next_arr_str = row['5.actual_arrival_next_station']
    next_station_arr = str_to_datetime(next_arr_str, format)
    curr_station_arr = str_to_datetime(curr_arr_str, format)
    avg_travel_time = row['average_predicted_travel_time']
    avg_dwell_time = row['average_predicted_dwell_time']

    departure_time = compute_departure_from_next(next_station_arr, avg_travel_time, format)
    if departure_time and departure_time < curr_arr_str:
        departure_time = compute_departure_from_current(curr_station_arr, avg_dwell_time, format)
        if departure_time and departure_time > next_arr_str:
            departure_time = None
    return departure_time

def process_condition_2_aa(row: pd.DataFrame, format: str)-> datetime:
    '''
    Process the arrival time for condition 2. This is when only previous station has data or current is terminating.

    Parameters:
    - row: DataFrame representing a row in the schedule.
    - format: Format of time strings.

    Returns:
    - Arrival time.
    '''
    prev_dep_str = row['3.actual_departure_prev_station']
    prev_station_dep = str_to_datetime(prev_dep_str, format)
    avg_travel_time = row['average_predicted_travel_time']
    return compute_arrival_from_prev(prev_station_dep, avg_travel_time, format)

def process_condition_2_ad(row: pd.DataFrame, format: str)-> datetime:
    '''
    Process the departure time for condition 2. This is when the next station 
    arrival has data and current station arrival has no data or is starting

    Parameters:
    - row: DataFrame representing a row in the schedule.
    - format: Format of time strings.

    Returns:
    - Departure time.
    '''
    next_arr_str = row['5.actual_arrival_next_station']
    next_station_arr = str_to_datetime(next_arr_str, format)
    avg_travel_time = row['average_predicted_travel_time']
    return compute_departure_from_next(next_station_arr, avg_travel_time, format)

def process_condition_3_aa(row: pd.DataFrame, format: str)-> datetime:
    '''
    Process the arrival time for condition 3. This is when only current station has data and it's not terminating.

    Parameters:
    - row: DataFrame representing a row in the schedule.
    - format: Format of time strings.

    Returns:
    - Arrival time.
    '''
    curr_dep_str = row['5.actual_departure_curr_station']
    curr_station_dep = str_to_datetime(curr_dep_str, format)
    avg_dwell_time = row['average_predicted_dwell_time']
    return compute_arrival_from_current(curr_station_dep, avg_dwell_time, format)

def process_condition_3_ad(row: pd.DataFrame, format: str)-> datetime:
    '''
    Process the departure time for condition 3. This is when the current station has data and is starting and next station arrival has no data.

    Parameters:
    - row: DataFrame representing a row in the schedule.
    - format: Format of time strings.

    Returns:
    - Departure time.
    '''
    curr_arr_str = row['3.actual_arrival_curr_station']
    curr_station_arr = str_to_datetime(curr_arr_str, format)
    avg_dwell_time = row['average_predicted_dwell_time']
    return compute_departure_from_current(curr_station_arr, avg_dwell_time, format)