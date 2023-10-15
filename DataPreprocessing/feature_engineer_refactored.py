import pandas as pd

from tqdm import tqdm

# Import from time_utils.py
from Utilities.datetime_utils import set_starting_terminating_times

# Import from schedule_utils.py
from Utilities.schedule_utils import calculate_missing_percentage
from Utilities.schedule_utils import calculate_metrics
from Utilities.schedule_utils import extract_schedule_OD_travel
from Utilities.schedule_utils import create_OD_pairs_dataframe
from Utilities.schedule_utils import drop_nan_pairs
from Utilities.schedule_utils import impute_missing_data
from Utilities.schedule_utils import assign_departure_order
from Utilities.schedule_utils import add_schedule_detail

# Import from dwell_time_utils.py
from Utilities.datetime_utils import dwell_time_extract
from Utilities.datetime_utils import dwell_time_summary_statistics
from Utilities.datetime_utils import travel_time_summary_statistics

# initialise tqdm for pandas
tqdm.pandas()

THRESHOLD_MINUTES = 500
MISSING_DATA_THRESHOLD_INITIAL = 10.0 
MISSING_DATA_THRESHOLD_FINAL = 0
STATION_OFFSET_TRAVEL_TIME = 1
STATION_OFFSET_DWELL_TIME = 2
NEXT_STATION_OFFSET = 1
FORMAT = '%H%M'

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
    historical_information = historical_information[historical_information['percentage_null'] <= MISSING_DATA_THRESHOLD_INITIAL].reset_index(drop=True)

    # Extract trips to find unique ones
    print('Extracting trips to find unique ones...')
    historical_information['locations'] = historical_information['5.schedule_detail'].apply(lambda x: x['location'].to_string())
    duplicate_list = historical_information.duplicated(['2.origin', '3.destination', '4.stops', 'locations'], keep='first').tolist()
    historical_information['duplicate_list'] = duplicate_list

    # Get unique trips
    print('Getting unique trips...')
    unique_trips = historical_information[~historical_information.duplicate_list]
    unique_trips = unique_trips.drop(columns=['1.date', 'percentage_null', 'duplicate_list']).sort_values(by='2.origin').reset_index(drop=True)

    # Feature engineer time-related metrics
    print('Feature engineer actual and predicted travel and dwell times...')
    historical_information['5.schedule_detail'] = historical_information['5.schedule_detail'].apply(calculate_metrics, args=(NEXT_STATION_OFFSET, STATION_OFFSET_TRAVEL_TIME, STATION_OFFSET_DWELL_TIME, THRESHOLD_MINUTES, FORMAT))
    
    # Extract dwell times for each station
    print('Extracting dwell times for each station...')
    dwell_time_stations, extreme_value_index = dwell_time_extract(historical_information, THRESHOLD_MINUTES)
    
    # Calculate summary statistics for dwell times
    print('Calculating summary statistics for dwell times...')
    station_dwell_time_unique = dwell_time_summary_statistics(dwell_time_stations)

    # Extract schedule origin, destination, travel time, and predicted travel time
    print('Extracting schedule origin, destination, travel time, and predicted travel time...')
    origins, destinations, travel_times, predicted_travel_times = extract_schedule_OD_travel(historical_information, STATION_OFFSET_TRAVEL_TIME)

    # Create OD pairs DataFrame
    print('Creating OD pairs DataFrame...')
    od_pairs = create_OD_pairs_dataframe(origins, destinations, travel_times, predicted_travel_times)

    # Calculate summary statistics for travel times
    print('Calculating summary statistics for travel times...')
    od_pairs_unique = travel_time_summary_statistics(od_pairs)

    # Drop rows with nan values
    historical_information = drop_nan_pairs(historical_information, od_pairs_unique, NEXT_STATION_OFFSET)
    
    # Impuete missing data
    print('Imputing missing data...')
    impute_missing_data(historical_information, od_pairs_unique, station_dwell_time_unique, NEXT_STATION_OFFSET, FORMAT)
    
    # Create OD pairs DataFrame
    print('Creating final OD pairs DataFrame...')
    od_pairs = create_OD_pairs_dataframe(origins, destinations, travel_times, predicted_travel_times)
    
    # Calculate summary statistics for travel times
    print('Calculating final summary statistics for travel times...')
    od_pairs_unique = travel_time_summary_statistics(od_pairs)
    
    # Extract dwell times for each station
    print('Extracting final dwell times for each station...')
    dwell_time_stations, extreme_value_index = dwell_time_extract(historical_information, THRESHOLD_MINUTES)

    # Calculate summary statistics for dwell times
    print('Calculating final summary statistics for dwell times...')
    station_dwell_time_unique = dwell_time_summary_statistics(dwell_time_stations)

    # Calculate missing data percentages
    print('Calculating final missing data percentages...')
    historical_information.loc[:,'percentage_null'] = historical_information['5.schedule_detail'].apply(calculate_missing_percentage)

    # Drop rows with more than any missing data
    print('Dropping final rows with more than any missing data...')
    historical_information = historical_information[historical_information['percentage_null'] <= MISSING_DATA_THRESHOLD_FINAL].reset_index(drop=True)

    # Assign departure order
    print('Assigning departure order...')
    historical_information = assign_departure_order(historical_information)

    return historical_information, station_dwell_time_unique, od_pairs_unique

if __name__ == '__main__':

    DATA_PATHS = ["Data/hist_info_DID_PAD_2016.csv"]
    OUTPUT_FILENAME = "Data/feature_engineered.csv"  # Adjust this accordingly, or include the dynamic filename generation logic.

    historical_information = pd.read_csv(DATA_PATHS[0])
    print("Adding schedule detail...")
    historical_information.loc[:,'5.schedule_detail'] = historical_information['5.schedule_detail'].progress_apply(add_schedule_detail)
    # process historical information to add actual and predicted travel time and dwell time and get unique trips
    historical_information_refactored = process_historical_data(historical_information)
    historical_information_refactored.to_csv(OUTPUT_FILENAME)

