# Purpose: To clean the data and extract the relevant information from the historical information data
import pandas as pd
import numpy as np
from tqdm import tqdm

def feature_engineer(historical_information):
    total_for_loops = 18

    #replace all starting and terminating station data arrival and departure times as 'starting' and 'terminating' 
    for i in tqdm(range(0,len(historical_information)), desc = '2/%d Giving arrival/departure time for starting/terminating stations' % total_for_loops):
        lenh = len(historical_information.loc[i,'5.schedule_detail'])
        (historical_information.iloc[i]['5.schedule_detail']).loc[0,'actual_ta']='starting'
        (historical_information.iloc[i]['5.schedule_detail']).loc[0,'gbtt_pta']='starting'
        (historical_information.iloc[i]['5.schedule_detail']).loc[lenh-1,'actual_td']='terminating'
        (historical_information.iloc[i]['5.schedule_detail']).loc[lenh-1,'gbtt_ptd']='terminating'


    # dropping all rows with greater than 10 percent missing data
    percentage_null = [None]*len(historical_information)

    for i in tqdm(range(0,len(historical_information)), desc = '3/%d Dropping all rows with greater than 10 percent missing data' % total_for_loops):
        x = historical_information.loc[i,'5.schedule_detail']
        x.loc[0:len(x)-1,'gbtt_ptd':'actual_ta']=x.loc[0:len(x)-1,'gbtt_ptd':'actual_ta'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
        y=x.loc[0:len(x)-1,'gbtt_ptd':'actual_ta'].shape
        percentage_null[i]=x.loc[0:len(x)-1,'gbtt_ptd':'actual_ta'].isnull().sum().sum()/(y[0]*y[1])*100

    historical_information['percentage_null'] = pd.Series(percentage_null, index=historical_information.index)
    historical_information = historical_information.drop(historical_information[historical_information.percentage_null > 10].index).reset_index()

    locations = [None]*len(historical_information)
    for i in tqdm(range(0,len(historical_information)), desc = '4/%d Extract all trips to find unique trips' % total_for_loops):
        locations[i] = historical_information.loc[i,'5.schedule_detail']['location'].to_string()
    historical_information['locations'] =  pd.Series(locations, index=historical_information.index)

    # gets list of unique trips
    duplicate_list = historical_information.duplicated(['2.origin','3.destination','4.stops','locations'], keep='first').tolist()
    historical_information['duplicate_list'] = pd.Series(duplicate_list, index=historical_information.index)

    unique_trips = historical_information[~historical_information.duplicate_list].drop(['index','1.date','percentage_null', 'duplicate_list'], axis=1).sort_values(by='2.origin').reset_index(drop =True)

    #calculating the time it takes to get from station a to b as well as arrival and departure delay
    import datetime
    from datetime import timedelta 
    for i in tqdm(range(0,len(historical_information)), desc = '5/%d Calculate time taken to travel and dwell time of all trips' % total_for_loops):
        x=historical_information.loc[i,'5.schedule_detail']
        travel_time = [0]*len(x)
        predicted_travel_time = [0]*len(x)
        actual_dwell_time = [0]*len(x)
        predicted_dwell_time = [0]*len(x)

        for j in range(0,len(travel_time)-1):
            if type(x.loc[j+1,'actual_ta'])==float:
                travel_time[j+1]=np.nan
                
            elif type(x.loc[j,'actual_td'])==float:
                travel_time[j+1]=np.nan
                
            else:
                format = '%H%M'
                start = x.loc[j,'actual_td']
                end = x.loc[j+1,'actual_ta']
                startDateTime = datetime.datetime.strptime(start, format)
                endDateTime = datetime.datetime.strptime(end, format)
                travel_time[j+1] = (endDateTime - startDateTime).total_seconds() / 60.0
                if np.absolute(travel_time[j+1])>500:
                    endDateTime = datetime.datetime.strptime(end, format) + timedelta(days=1)
                    travel_time[j+1] = (endDateTime - startDateTime).total_seconds() / 60.0
        
        x['travel_time'] = pd.Series(travel_time, index=x.index)  
        historical_information.loc[i,'5.schedule_detail']['travel_time']=x['travel_time']
        
        for j in range(0,len(predicted_travel_time)-1):
            if type(x.loc[j+1,'gbtt_pta'])==float:
                predicted_travel_time[j+1]=np.nan
                
            elif type(x.loc[j,'gbtt_ptd'])==float:
                predicted_travel_time[j+1]=np.nan
                
            else:
                format = '%H%M'
                start = x.loc[j,'gbtt_ptd']
                end = x.loc[j+1,'gbtt_pta']
                startDateTime = datetime.datetime.strptime(start, format)
                endDateTime = datetime.datetime.strptime(end, format)
                predicted_travel_time[j+1] = (endDateTime - startDateTime).total_seconds() / 60.0
                if np.absolute(predicted_travel_time[j+1])>500:
                    endDateTime = datetime.datetime.strptime(end, format) + timedelta(days=1)
                    predicted_travel_time[j+1] = (endDateTime - startDateTime).total_seconds() / 60.0
        
        x['travel_time_predicted'] = pd.Series(predicted_travel_time, index=x.index)  
        historical_information.loc[i,'5.schedule_detail']['travel_time_predicted']=x['travel_time_predicted']
        
        for j in range(0,len(actual_dwell_time)-2):
            if (type(x.loc[j+1,'actual_ta'])==float) or (type(x.loc[j+1,'actual_td'])==float):
                actual_dwell_time[j+1]=np.nan
            
            elif (type(x.loc[j+1,'gbtt_pta'])==float) or (type(x.loc[j+1,'gbtt_ptd'])==float):
                predicted_dwell_time[j+1]=np.nan
            
            else:
                format = '%H%M'
                
                arrival = x.loc[j+1,'actual_ta']
                departure = x.loc[j+1,'actual_td']
                arrivalDateTime = datetime.datetime.strptime(arrival, format)
                departureDateTime = datetime.datetime.strptime(departure, format)
                actual_dwell_time[j+1] = (departureDateTime - arrivalDateTime).total_seconds() / 60.0
                
                arrival_predicted = x.loc[j+1,'gbtt_pta']
                departure_predicted = x.loc[j+1,'gbtt_ptd']
                arrivalDateTimePredicted = datetime.datetime.strptime(arrival_predicted, format)
                departureDateTimePredicted = datetime.datetime.strptime(departure_predicted, format)
                predicted_dwell_time[j+1] = (departureDateTimePredicted - arrivalDateTimePredicted).total_seconds() / 60.0
                
                if np.absolute(actual_dwell_time[j+1])>500:
                    departureDateTime = datetime.datetime.strptime(departure, format) + timedelta(days=1)
                    actual_dwell_time[j+1] = (departureDateTime - arrivalDateTime).total_seconds() / 60.0
                
                if np.absolute(predicted_dwell_time[j+1])>500:
                    departureDateTimePredicted = datetime.datetime.strptime(departure_predicted, format) + timedelta(days=1)
                    predicted_dwell_time[j+1] = (departureDateTimePredicted - arrivalDateTimePredicted).total_seconds() / 60.0
        
        x['dwell_time'] = pd.Series(actual_dwell_time, index=x.index)  
        historical_information.loc[i,'5.schedule_detail']['dwell_time']=x['dwell_time']
        x['dwell_time_predicted'] = pd.Series(predicted_dwell_time, index=x.index)  
        historical_information.loc[i,'5.schedule_detail']['dwell_time_predicted']=x['dwell_time_predicted']

    #calculating average dwell times
    no_stations = sum(historical_information['4.stops'])
    all_stations = [None]*no_stations
    dwell_time_all_stations = [None]*no_stations
    dwell_time_predicted_all_stations = [None]*no_stations
    k=0
    extreme_value_index = []
    for i in tqdm(range(0,len(historical_information)), desc = '6/%d Extracting all dwell times' % total_for_loops):
        x = historical_information.loc[i,'5.schedule_detail']
        for j in range(0,len(x)):
            all_stations[k]=x.loc[j,'location']
            dwell_time_all_stations[k]=x.loc[j,'dwell_time']
            dwell_time_predicted_all_stations[k] = x.loc[j, 'dwell_time_predicted']
            if (np.absolute(dwell_time_all_stations[k])>500) or (np.absolute(dwell_time_predicted_all_stations[k])>500) :
                extreme_value_index.append(i)
            k=k+1

    dwell_time_stations = pd.DataFrame({'1.station':all_stations,'2.dwell_time':dwell_time_all_stations,'3.dwell_time_predicted':dwell_time_predicted_all_stations})

    #find unique stations
    stations_unique = dwell_time_stations
    stations_duplicate_list = stations_unique.duplicated('1.station', keep='first').tolist()
    stations_unique['duplicate_list'] = pd.Series(stations_duplicate_list, index=stations_unique.index)
    stations_unique = stations_unique[~stations_unique.duplicate_list].drop(['duplicate_list'], axis=1).reset_index(drop = True)
    stations_unique = stations_unique.drop(columns='2.dwell_time').drop(columns='3.dwell_time_predicted')

    #calculating total and average dwell times
    total_dwell_time_unique = [0]*len(stations_unique)
    average_dwell_time_unique = [0]*len(stations_unique)
    total_dwell_time_unique_predicted = [0]*len(stations_unique)
    average_dwell_time_unique_predicted = [0]*len(stations_unique)
    station_interested = [0]*len(stations_unique)

    for i in tqdm(range(0,len(stations_unique)), desc = '7/%d calculating average and total dwell times' % total_for_loops):
        station = stations_unique.loc[i,'1.station']
        df = dwell_time_stations[dwell_time_stations['1.station']==station]
        total_dwell_time = df['2.dwell_time'].sum() 
        total_dwell_time_predicted = df['3.dwell_time_predicted'].sum() 
        average_dwell_time = total_dwell_time/len(df)
        average_dwell_time_predicted = total_dwell_time_predicted/len(df)
        
        station_interested[i] = station
        total_dwell_time_unique[i] = total_dwell_time
        average_dwell_time_unique[i] = average_dwell_time
        total_dwell_time_unique_predicted[i] = total_dwell_time_predicted
        average_dwell_time_unique_predicted[i] = average_dwell_time_predicted

    station_dwell_time_unique = pd.DataFrame ({'1.station':station_interested,
                                            '2.total_dwell_time':total_dwell_time_unique,
                                            '3.total_dwell_time_predicted':total_dwell_time_unique_predicted,
                                            '4.average_dwell_time':average_dwell_time_unique,
                                            '5.average_dwell_time_predicted':average_dwell_time_unique_predicted
                                            })
    station_dwell_time_unique = station_dwell_time_unique.loc[~(station_dwell_time_unique==0).all(axis=1)] #drop rows where all data is 0

    #finding unique OD pairs
    no_station_pairs = sum(historical_information['4.stops'])-len(historical_information)
    all_origin_stations=[None]*no_station_pairs
    all_destination_stations = [None]*no_station_pairs
    time_travelled_all_OD = [None]*no_station_pairs
    time_travelled_all_OD_predicted = [None]*no_station_pairs
    k=0
    extreme_value_index = []
    for i in tqdm(range(0,len(historical_information)), desc = '8/%d Find unique OD pairs' % total_for_loops):
        x = historical_information.loc[i,'5.schedule_detail']
        origin_stations = x.loc[0:len(x)-2,'location'].reset_index(drop = True)
        destination_stations = x.loc[1:len(x)-1,'location'].reset_index(drop = True)
        time_travelled = x.loc[1:len(x)-1,'travel_time'].reset_index(drop = True)
        time_travelled_predicted = x.loc[1:len(x)-1,'travel_time_predicted'].reset_index(drop = True)
        for j in range(0,len(origin_stations)):
            all_origin_stations[k]=origin_stations.loc[j]
            all_destination_stations[k]=destination_stations[j]
            time_travelled_all_OD[k] = time_travelled[j]
            time_travelled_all_OD_predicted[k] = time_travelled_predicted[j]
            if (np.absolute(time_travelled_all_OD[k])>500) or (np.absolute(time_travelled_all_OD_predicted[k])>500) :
                extreme_value_index.append(i)
            k=k+1

    OD_pairs = pd.DataFrame({'1.origin': all_origin_stations,'2.destination' : all_destination_stations, '3.time_travelled' : time_travelled_all_OD,'4.predicted_time_travelled' : time_travelled_all_OD_predicted})
    OD_pairs_original = OD_pairs

    #find unique pairs
    OD_pairs_unique = OD_pairs
    OD_duplicate_list = OD_pairs_unique.duplicated(['1.origin','2.destination'], keep='first').tolist()
    OD_pairs_unique['duplicate_list'] = pd.Series(OD_duplicate_list, index=OD_pairs_unique.index)
    OD_pairs_unique = OD_pairs_unique[~OD_pairs_unique.duplicate_list].drop(['duplicate_list'], axis=1).reset_index(drop = True)
    OD_pairs_unique = OD_pairs_unique.drop(columns='3.time_travelled').drop(columns='4.predicted_time_travelled')

    #get rid of nan rows
    OD_pairs = OD_pairs[np.isfinite(OD_pairs['3.time_travelled'])]
    OD_pairs = OD_pairs[np.isfinite(OD_pairs['4.predicted_time_travelled'])]
    total_time_travelled_unique_OD = [None]*len(OD_pairs_unique)
    average_time_travelled_unique_OD = [None]*len(OD_pairs_unique)
    total_time_travelled_unique_OD_predicted = [None]*len(OD_pairs_unique)
    average_time_travelled_unique_OD_predicted = [None]*len(OD_pairs_unique)

    #Finding average time travelled for each OD pair
    for i in tqdm(range(0,len(OD_pairs_unique)), desc = '9/%d Find average time for each OD pair' % total_for_loops):
        origin_station = OD_pairs_unique.loc[i,'1.origin']
        destination_station = OD_pairs_unique.loc[i,'2.destination']
        df = OD_pairs[(OD_pairs['1.origin']==origin_station) & (OD_pairs['2.destination']==destination_station)]
        total_time_travelled = df['3.time_travelled'].sum() 
        total_time_travelled_predicted = df['4.predicted_time_travelled'].sum() 
        average_time_travelled = total_time_travelled/len(df)
        average_time_travelled_predicted = total_time_travelled_predicted/len(df)
        total_time_travelled_unique_OD[i] = total_time_travelled
        average_time_travelled_unique_OD[i] = average_time_travelled
        total_time_travelled_unique_OD_predicted[i] = total_time_travelled_predicted
        average_time_travelled_unique_OD_predicted[i] = average_time_travelled_predicted
        
    OD_pairs_unique['total_travel_time'] = pd.Series(total_time_travelled_unique_OD, index=OD_pairs_unique.index)
    OD_pairs_unique['average_travel_time'] = pd.Series(average_time_travelled_unique_OD, index=OD_pairs_unique.index)
    OD_pairs_unique['total_travel_time_predicted'] = pd.Series(total_time_travelled_unique_OD_predicted, index=OD_pairs_unique.index)
    OD_pairs_unique['average_travel_time_predicted'] = pd.Series(average_time_travelled_unique_OD_predicted, index=OD_pairs_unique.index)

    #get list of all nan OD pairs this is so that they can be removed from historical_information data
    OD_pairs_unique_nan = OD_pairs_unique[(np.isnan(OD_pairs_unique['average_travel_time']))].reset_index(drop = True)

    dropped = [0]*len(historical_information)
    m=0
    #Drop all historical information points that have the nan pairs
        
    for i in tqdm(range(0,len(historical_information)), desc = '10/%d Dropping all rows with no information' % total_for_loops):
        x = historical_information.loc[i,'5.schedule_detail']
        
        for j in range(0,len(OD_pairs_unique_nan)):
            origin_station_drop = OD_pairs_unique_nan.loc[j,'1.origin']
            destination_station_drop = OD_pairs_unique_nan.loc[j,'2.destination']
        
            for k in range(0,len(x)-1):
                origin_station = x.loc[k,'location']
                destination_station = x.loc[k+1,'location']
                
                if (origin_station == origin_station_drop) and (destination_station == destination_station_drop):
                    dropped[m] = 1
        m=m+1   
        
    historical_information['dropped'] = pd.Series(dropped, index=historical_information.index)
    historical_information = historical_information.drop(historical_information[historical_information.dropped == 1].index).reset_index(drop = True)

    total_null = 0
    total_null_new = 1
    #Extracting missing information
    while total_null != total_null_new:
        
        total_null = 0
        total_actual_arrival_null = 0
        total_actual_departure_null = 0
        total_predicted_arrival_null = 0
        total_predicted_departure_null = 0
        
        for i in tqdm(range(0,len(historical_information)), desc = '11/%d Calculating the total number of nan values' % total_for_loops):
            x = historical_information.loc[i,'5.schedule_detail']
            total_null= total_null + x.loc[0:len(x)-1,'actual_ta':'gbtt_ptd'].isnull().sum().sum()
            total_actual_arrival_null = total_actual_arrival_null + x.loc[0:len(x)-1,'actual_ta'].isnull().sum().sum()
            total_actual_departure_null = total_actual_departure_null + x.loc[0:len(x)-1,'actual_td'].isnull().sum().sum()
            total_predicted_arrival_null = total_predicted_arrival_null + x.loc[0:len(x)-1,'gbtt_pta'].isnull().sum().sum()
            total_predicted_departure_null = total_predicted_departure_null + x.loc[0:len(x)-1,'gbtt_ptd'].isnull().sum().sum()
        
        print('Total null values =',total_null)

        #extracting actual arrival nans
        aa_origin_station_nan = [None]*total_actual_arrival_null
        aa_destination_station_nan = [None]*total_actual_arrival_null
        aa_actual_arrival_nan = [None]*total_actual_arrival_null 
        aa_actual_departure_origin_nan = [None]*total_actual_arrival_null 
        aa_actual_departure_destination_nan = [None]*total_actual_arrival_null 
        aa_historical_information_index_nan = [None]*total_actual_arrival_null
        aa_schedule_detail_index_nan = [None]*total_actual_arrival_null
        
        k=0
        
        for i in tqdm(range(0,len(historical_information)), desc = '12/%d Extracting missing information' % total_for_loops):
            x=historical_information.loc[i,'5.schedule_detail']
            for j in range(0,len(x)-1):
                if type(x.loc[j+1,'actual_ta'])==float:
                    aa_origin_station_nan[k] = x.loc[j,'location']
                    aa_destination_station_nan[k] = x.loc[j+1,'location']
                    aa_actual_arrival_nan[k] = x.loc[j+1,'actual_ta']
                    aa_actual_departure_origin_nan[k] = x.loc[j,'actual_td']
                    aa_actual_departure_destination_nan[k] = x.loc[j+1,'actual_td']
                    aa_historical_information_index_nan[k] = i
                    aa_schedule_detail_index_nan[k] = j+1
                    k=k+1
                    
        aa_nan = pd.DataFrame({'1.origin': aa_origin_station_nan,
                            '2.destination':aa_destination_station_nan,
                            '3.actual_departure_prev_station':aa_actual_departure_origin_nan,
                            '4.actual_arrival':aa_actual_arrival_nan,
                            '5.actual_departure_current_station':aa_actual_departure_destination_nan,
                            '6.h_i_index':aa_historical_information_index_nan,
                            '7.s_d_index':aa_schedule_detail_index_nan})
        
        aa_nan = aa_nan.dropna(axis=0, how='all') #drop rows where all data is null
        
        #isolate data points with all three missing datapoints in org departure, des arrival and des dep
        average_actual_travel_time = [0]*len(aa_nan)
        average_predicted_travel_time = [0]*len(aa_nan)
        average_dwell_time = [0]*len(aa_nan)
        average_predicted_dwell_time = [0]*len(aa_nan)
        aa_nan['average_actual_travel_time'] = pd.Series(average_actual_travel_time, index=aa_nan.index)
        aa_nan['average_predicted_travel_time'] = pd.Series(average_predicted_travel_time, index=aa_nan.index)
        aa_nan['average_dwell_time'] = pd.Series(average_dwell_time, index=aa_nan.index)
        aa_nan['average_predicted_dwell_time'] = pd.Series(average_predicted_dwell_time, index=aa_nan.index)
        
        for i in tqdm(range(0,len(aa_nan)), desc = '13/%d Inputting actual and predicted travel time and dwell time' % total_for_loops):
            origin_station = aa_nan.loc[i,'1.origin']
            destination_station = aa_nan.loc[i,'2.destination']
            
            for j in range(0,len(OD_pairs_unique)):
                if (OD_pairs_unique.loc[j,'1.origin'] == origin_station) and (OD_pairs_unique.loc[j,'2.destination'] == destination_station):
                    
                    aa_nan.loc[i,'average_actual_travel_time'] = OD_pairs_unique.loc[j,'average_travel_time']
                    aa_nan.loc[i,'average_predicted_travel_time'] = OD_pairs_unique.loc[j,'average_travel_time_predicted']
            
            for j in range(0,len(station_dwell_time_unique)):
                if destination_station == station_dwell_time_unique.loc[j,'1.station']:
                    
                    aa_nan.loc[i,'average_dwell_time'] = station_dwell_time_unique.loc[j,'4.average_dwell_time']
                    aa_nan.loc[i,'average_predicted_dwell_time'] = station_dwell_time_unique.loc[j,'5.average_dwell_time_predicted']
                    
        
        #filling in missing data for actual arrival time 
        actual_arrival_time_1 = [0]*len(aa_nan)
        aa_nan['actual_arrival_time_1'] = pd.Series(actual_arrival_time_1, index=aa_nan.index)
        for i in tqdm(range(0,len(aa_nan)), desc = '14/%d filling in missing data for actual_arrival_time' % total_for_loops):
            if (type(aa_nan.loc[i,'3.actual_departure_prev_station']) != float) and (type(aa_nan.loc[i,'5.actual_departure_current_station']) != float) and (aa_nan.loc[i,'5.actual_departure_current_station'] != 'terminating'):
                prev_station_dep = datetime.datetime.strptime(aa_nan.loc[i,'3.actual_departure_prev_station'], format)
                aa_nan.loc[i,'actual_arrival_time_1'] = (prev_station_dep + timedelta(minutes=aa_nan.loc[i,'average_predicted_travel_time'])).strftime(format)
                
                if aa_nan.loc[i,'actual_arrival_time_1'] > aa_nan.loc[i,'5.actual_departure_current_station']:
                    curr_station_dep = datetime.datetime.strptime(aa_nan.loc[i,'5.actual_departure_current_station'], format)
                    aa_nan.loc[i,'actual_arrival_time_1'] = (curr_station_dep - timedelta(minutes=aa_nan.loc[i,'average_predicted_dwell_time'])).strftime(format)
                    
                    if aa_nan.loc[i,'actual_arrival_time_1'] < aa_nan.loc[i,'3.actual_departure_prev_station']:
                        aa_nan.loc[i,'actual_arrival_time_1'] = np.nan
                        
                aa_nan.loc[i,'4.actual_arrival'] = aa_nan.loc[i,'actual_arrival_time_1']
                (historical_information.loc[aa_nan.loc[i,'6.h_i_index'],'5.schedule_detail']).loc[aa_nan.loc[i,'7.s_d_index'],'actual_ta'] = aa_nan.loc[i,'actual_arrival_time_1']
        
            if (type(aa_nan.loc[i,'3.actual_departure_prev_station']) != float) and ((type(aa_nan.loc[i,'5.actual_departure_current_station']) == float) or (aa_nan.loc[i,'5.actual_departure_current_station'] == 'terminating')):
                prev_station_dep = datetime.datetime.strptime(aa_nan.loc[i,'3.actual_departure_prev_station'], format)
                aa_nan.loc[i,'actual_arrival_time_1'] = (prev_station_dep + timedelta(minutes=aa_nan.loc[i,'average_predicted_travel_time'])).strftime(format)
                        
                aa_nan.loc[i,'4.actual_arrival'] = aa_nan.loc[i,'actual_arrival_time_1']
                (historical_information.loc[aa_nan.loc[i,'6.h_i_index'],'5.schedule_detail']).loc[aa_nan.loc[i,'7.s_d_index'],'actual_ta'] = aa_nan.loc[i,'actual_arrival_time_1']
        
            if (type(aa_nan.loc[i,'3.actual_departure_prev_station']) == float) and (type(aa_nan.loc[i,'5.actual_departure_current_station']) != float) and (aa_nan.loc[i,'5.actual_departure_current_station'] != 'terminating'):
                curr_station_dep = datetime.datetime.strptime(aa_nan.loc[i,'5.actual_departure_current_station'], format)
                aa_nan.loc[i,'actual_arrival_time_1'] = (curr_station_dep - timedelta(minutes=aa_nan.loc[i,'average_predicted_dwell_time'])).strftime(format)
                        
                aa_nan.loc[i,'4.actual_arrival'] = aa_nan.loc[i,'actual_arrival_time_1']
                (historical_information.loc[aa_nan.loc[i,'6.h_i_index'],'5.schedule_detail']).loc[aa_nan.loc[i,'7.s_d_index'],'actual_ta'] = aa_nan.loc[i,'actual_arrival_time_1']
        
        #extracting actual departure nans
        ad_curr_station_nan = [None]*total_actual_departure_null
        ad_next_station_nan = [None]*total_actual_departure_null
        ad_actual_departure_curr_nan = [None]*total_actual_departure_null 
        ad_actual_arrival_curr_nan = [None]*total_actual_departure_null 
        ad_actual_arrival_next_nan = [None]*total_actual_departure_null 
        ad_historical_information_index_nan = [None]*total_actual_departure_null
        ad_schedule_detail_index_nan = [None]*total_actual_departure_null
        
        k=0
        
        for i in tqdm(range(0,len(historical_information)), desc = '15/%d Extracting missing departure information' % total_for_loops):
            x=historical_information.loc[i,'5.schedule_detail']
            for j in range(0,len(x)-1):
                if type(x.loc[j,'actual_td'])==float:
                    ad_curr_station_nan[k] = x.loc[j,'location']
                    ad_next_station_nan[k] = x.loc[j+1,'location']
                    ad_actual_departure_curr_nan[k] = x.loc[j,'actual_td']
                    ad_actual_arrival_curr_nan[k] = x.loc[j,'actual_ta']
                    ad_actual_arrival_next_nan[k] = x.loc[j+1,'actual_ta']
                    ad_historical_information_index_nan[k] = i
                    ad_schedule_detail_index_nan[k] = j
                    k=k+1
                    
        ad_nan = pd.DataFrame({'1.current_station': ad_curr_station_nan,
                            '2.next_station':ad_next_station_nan,
                            '3.actual_arrival_curr_station':ad_actual_arrival_curr_nan,
                            '4.actual_departure_curr_station':ad_actual_departure_curr_nan,
                            '5.actual_arrival_next_station':ad_actual_arrival_next_nan,
                            '6.h_i_index':ad_historical_information_index_nan,
                            '7.s_d_index':ad_schedule_detail_index_nan})
        
        ad_nan = ad_nan.dropna(axis=0, how='all') #drop rows where all data is null
        
        #inputting predicted travel and dwell time
        average_actual_travel_time = [0]*len(ad_nan)
        average_predicted_travel_time = [0]*len(ad_nan)
        average_dwell_time = [0]*len(ad_nan)
        average_predicted_dwell_time = [0]*len(ad_nan)
        ad_nan['average_actual_travel_time'] = pd.Series(average_actual_travel_time, index=ad_nan.index)
        ad_nan['average_predicted_travel_time'] = pd.Series(average_predicted_travel_time, index=ad_nan.index)
        ad_nan['average_dwell_time'] = pd.Series(average_dwell_time, index=ad_nan.index)
        ad_nan['average_predicted_dwell_time'] = pd.Series(average_predicted_dwell_time, index=ad_nan.index)
        
        for i in tqdm(range(0,len(ad_nan)), desc = '16/%d Inputting actual and predicted travel time and dwell time' % total_for_loops):
            curr_station = ad_nan.loc[i,'1.current_station']
            next_station = ad_nan.loc[i,'2.next_station']
            
            for j in range(0,len(OD_pairs_unique)):
                if (curr_station == OD_pairs_unique.loc[j,'1.origin']) and (next_station==OD_pairs_unique.loc[j,'2.destination']):
                    
                    ad_nan.loc[i,'average_actual_travel_time'] = OD_pairs_unique.loc[j,'average_travel_time']
                    ad_nan.loc[i,'average_predicted_travel_time'] = OD_pairs_unique.loc[j,'average_travel_time_predicted']
            
            for k in range(0,len(station_dwell_time_unique)):
                if curr_station == station_dwell_time_unique.loc[k,'1.station']:
                    
                    ad_nan.loc[i,'average_dwell_time'] = station_dwell_time_unique.loc[k,'4.average_dwell_time']
                    ad_nan.loc[i,'average_predicted_dwell_time'] = station_dwell_time_unique.loc[k,'5.average_dwell_time_predicted']
        
        #filling in missing data for actual departure time 
        for i in tqdm(range(0,len(ad_nan)), desc = '17/%d filling in missing data for actual_arrival_time' % total_for_loops):
            if (type(ad_nan.loc[i,'3.actual_arrival_curr_station']) != float) and (type(ad_nan.loc[i,'5.actual_arrival_next_station']) != float) and (ad_nan.loc[i,'3.actual_arrival_curr_station'] != 'starting'):
                next_station_arr = datetime.datetime.strptime(ad_nan.loc[i,'5.actual_arrival_next_station'], format)
                ad_nan.loc[i,'4.actual_departure_curr_station'] = (next_station_arr - timedelta(minutes=ad_nan.loc[i,'average_predicted_travel_time'])).strftime(format)
                
                if ad_nan.loc[i,'4.actual_departure_curr_station'] < ad_nan.loc[i,'3.actual_arrival_curr_station']:
                    curr_station_arr = datetime.datetime.strptime(ad_nan.loc[i,'3.actual_arrival_curr_station'], format)
                    ad_nan.loc[i,'4.actual_departure_curr_station'] = (curr_station_arr + timedelta(minutes=ad_nan.loc[i,'average_predicted_dwell_time'])).strftime(format)
                    
                    if ad_nan.loc[i,'4.actual_departure_curr_station'] > ad_nan.loc[i,'5.actual_arrival_next_station']:
                        ad_nan.loc[i,'4.actual_departure_curr_station'] = np.nan
                        
                (historical_information.loc[ad_nan.loc[i,'6.h_i_index'],'5.schedule_detail']).loc[ad_nan.loc[i,'7.s_d_index'],'actual_td'] = ad_nan.loc[i,'4.actual_departure_curr_station']
            
            if ((ad_nan.loc[i,'3.actual_arrival_curr_station']=='starting') and (type(ad_nan.loc[i,'5.actual_arrival_next_station'])!=float)) or ((type(ad_nan.loc[i,'3.actual_arrival_curr_station'])==float) and (type(ad_nan.loc[i,'5.actual_arrival_next_station'])!=float))  :
                next_station_arr = datetime.datetime.strptime(ad_nan.loc[i,'5.actual_arrival_next_station'], format)
                ad_nan.loc[i,'4.actual_departure_curr_station'] = (next_station_arr - timedelta(minutes=ad_nan.loc[i,'average_predicted_travel_time'])).strftime(format)
                (historical_information.loc[ad_nan.loc[i,'6.h_i_index'],'5.schedule_detail']).loc[ad_nan.loc[i,'7.s_d_index'],'actual_td'] = ad_nan.loc[i,'4.actual_departure_curr_station']    
                
            if (type(ad_nan.loc[i,'3.actual_arrival_curr_station'])!=float) and (type(ad_nan.loc[i,'5.actual_arrival_next_station'])==float) and (ad_nan.loc[i,'3.actual_arrival_curr_station'] != 'starting'):
                curr_station_arr = datetime.datetime.strptime(ad_nan.loc[i,'3.actual_arrival_curr_station'], format)
                ad_nan.loc[i,'4.actual_departure_curr_station'] = (curr_station_arr + timedelta(minutes=ad_nan.loc[i,'average_predicted_dwell_time'])).strftime(format)
                (historical_information.loc[ad_nan.loc[i,'6.h_i_index'],'5.schedule_detail']).loc[ad_nan.loc[i,'7.s_d_index'],'actual_td'] = ad_nan.loc[i,'4.actual_departure_curr_station']    
                    
        total_null_new = 0
        
        
        for i in tqdm(range(0,len(historical_information)), desc = '11/%d Calculating the total number of nan values' % total_for_loops):
            x = historical_information.loc[i,'5.schedule_detail']
            total_null_new = total_null_new + x.loc[0:len(x)-1,'actual_ta':'gbtt_ptd'].isnull().sum().sum()
            
        print('Total new null values =',total_null_new)
                                                        
    #these methods shall be repeated until no more data can be injected
            
    #after certain loops remove all historical info datapoints that have missing data

    #finding unique OD pairs
    no_station_pairs = sum(historical_information['4.stops'])-len(historical_information)
    all_origin_stations=[None]*no_station_pairs
    all_destination_stations = [None]*no_station_pairs
    time_travelled_all_OD = [None]*no_station_pairs
    time_travelled_all_OD_predicted = [None]*no_station_pairs
    k=0
    extreme_value_index = []
    for i in tqdm(range(0,len(historical_information)), desc = '8/%d Find unique OD pairs' % total_for_loops):
        x = historical_information.loc[i,'5.schedule_detail']
        origin_stations = x.loc[0:len(x)-2,'location'].reset_index(drop = True)
        destination_stations = x.loc[1:len(x)-1,'location'].reset_index(drop = True)
        time_travelled = x.loc[1:len(x)-1,'travel_time'].reset_index(drop = True)
        time_travelled_predicted = x.loc[1:len(x)-1,'travel_time_predicted'].reset_index(drop = True)
        for j in range(0,len(origin_stations)):
            all_origin_stations[k]=origin_stations[j]
            all_destination_stations[k]=destination_stations[j]
            time_travelled_all_OD[k] = time_travelled[j]
            time_travelled_all_OD_predicted[k] = time_travelled_predicted[j]
            if (np.absolute(time_travelled_all_OD[k])>500) or (np.absolute(time_travelled_all_OD_predicted[k])>500) :
                extreme_value_index.append(i)
            k=k+1

    OD_pairs = pd.DataFrame({'1.origin': all_origin_stations,'2.destination' : all_destination_stations, '3.time_travelled' : time_travelled_all_OD,'4.predicted_time_travelled' : time_travelled_all_OD_predicted})
    OD_pairs_original = OD_pairs

    #find unique pairs
    OD_pairs_unique = OD_pairs
    OD_duplicate_list = OD_pairs_unique.duplicated(['1.origin','2.destination'], keep='first').tolist()
    OD_pairs_unique['duplicate_list'] = pd.Series(OD_duplicate_list, index=OD_pairs_unique.index)
    OD_pairs_unique = OD_pairs_unique[~OD_pairs_unique.duplicate_list].drop(['duplicate_list'], axis=1).reset_index(drop = True)
    OD_pairs_unique = OD_pairs_unique.drop(columns='3.time_travelled').drop(columns='4.predicted_time_travelled')

    #get rid of nan rows
    OD_pairs = OD_pairs[np.isfinite(OD_pairs['3.time_travelled'])]
    OD_pairs = OD_pairs[np.isfinite(OD_pairs['4.predicted_time_travelled'])]
    total_time_travelled_unique_OD = [None]*len(OD_pairs_unique)
    average_time_travelled_unique_OD = [None]*len(OD_pairs_unique)
    total_time_travelled_unique_OD_predicted = [None]*len(OD_pairs_unique)
    average_time_travelled_unique_OD_predicted = [None]*len(OD_pairs_unique)

    #Finding average time travelled for each OD pair
    for i in tqdm(range(0,len(OD_pairs_unique)), desc = '9/%d Find average time for each OD pair' % total_for_loops):
        origin_station = OD_pairs_unique.loc[i,'1.origin']
        destination_station = OD_pairs_unique.loc[i,'2.destination']
        df = OD_pairs[(OD_pairs['1.origin']==origin_station) & (OD_pairs['2.destination']==destination_station)]
        total_time_travelled = df['3.time_travelled'].sum() 
        total_time_travelled_predicted = df['4.predicted_time_travelled'].sum() 
        average_time_travelled = total_time_travelled/len(df)
        average_time_travelled_predicted = total_time_travelled_predicted/len(df)
        total_time_travelled_unique_OD[i] = total_time_travelled
        average_time_travelled_unique_OD[i] = average_time_travelled
        total_time_travelled_unique_OD_predicted[i] = total_time_travelled_predicted
        average_time_travelled_unique_OD_predicted[i] = average_time_travelled_predicted
        
    OD_pairs_unique['total_travel_time'] = pd.Series(total_time_travelled_unique_OD, index=OD_pairs_unique.index)
    OD_pairs_unique['average_travel_time'] = pd.Series(average_time_travelled_unique_OD, index=OD_pairs_unique.index)
    OD_pairs_unique['total_travel_time_predicted'] = pd.Series(total_time_travelled_unique_OD_predicted, index=OD_pairs_unique.index)
    OD_pairs_unique['average_travel_time_predicted'] = pd.Series(average_time_travelled_unique_OD_predicted, index=OD_pairs_unique.index)

    #calculating average dwell times
    no_stations = sum(historical_information['4.stops'])
    all_stations = [None]*no_stations
    dwell_time_all_stations = [None]*no_stations
    dwell_time_predicted_all_stations = [None]*no_stations
    k=0
    extreme_value_index = []
    for i in tqdm(range(0,len(historical_information)), desc = '6/%d Extracting all dwell times' % total_for_loops):
        x = historical_information.loc[i,'5.schedule_detail']
        for j in range(0,len(x)):
            all_stations[k]=x.loc[j,'location']
            dwell_time_all_stations[k]=x.loc[j,'dwell_time']
            dwell_time_predicted_all_stations[k] = x.loc[j, 'dwell_time_predicted']
            if (np.absolute(dwell_time_all_stations[k])>500) or (np.absolute(dwell_time_predicted_all_stations[k])>500) :
                extreme_value_index.append(i)
            k=k+1

    dwell_time_stations = pd.DataFrame({'1.station':all_stations,'2.dwell_time':dwell_time_all_stations,'3.dwell_time_predicted':dwell_time_predicted_all_stations})

    #find unique stations
    stations_unique = dwell_time_stations
    stations_duplicate_list = stations_unique.duplicated('1.station', keep='first').tolist()
    stations_unique['duplicate_list'] = pd.Series(stations_duplicate_list, index=stations_unique.index)
    stations_unique = stations_unique[~stations_unique.duplicate_list].drop(['duplicate_list'], axis=1).reset_index(drop = True)
    stations_unique = stations_unique.drop(columns='2.dwell_time').drop(columns='3.dwell_time_predicted')

    #calculating total and average dwell times
    total_dwell_time_unique = [0]*len(stations_unique)
    average_dwell_time_unique = [0]*len(stations_unique)
    total_dwell_time_unique_predicted = [0]*len(stations_unique)
    average_dwell_time_unique_predicted = [0]*len(stations_unique)
    station_interested = [0]*len(stations_unique)

    for i in tqdm(range(0,len(stations_unique)), desc = '7/%d calculating average and total dwell times' % total_for_loops):
        station = stations_unique.loc[i,'1.station']
        df = dwell_time_stations[dwell_time_stations['1.station']==station]
        total_dwell_time = df['2.dwell_time'].sum() 
        total_dwell_time_predicted = df['3.dwell_time_predicted'].sum() 
        average_dwell_time = total_dwell_time/len(df)
        average_dwell_time_predicted = total_dwell_time_predicted/len(df)
        
        station_interested[i] = station
        total_dwell_time_unique[i] = total_dwell_time
        average_dwell_time_unique[i] = average_dwell_time
        total_dwell_time_unique_predicted[i] = total_dwell_time_predicted
        average_dwell_time_unique_predicted[i] = average_dwell_time_predicted

    station_dwell_time_unique = pd.DataFrame ({'1.station':station_interested,
                                            '2.total_dwell_time':total_dwell_time_unique,
                                            '3.total_dwell_time_predicted':total_dwell_time_unique_predicted,
                                            '4.average_dwell_time':average_dwell_time_unique,
                                            '5.average_dwell_time_predicted':average_dwell_time_unique_predicted
                                            })

    # dropping all rows with missing data
    percentage_null = [None]*len(historical_information)
    historical_information = historical_information.drop(columns = 'index')
    historical_information = historical_information.drop(columns = 'percentage_null')

    for i in tqdm(range(0,len(historical_information)), desc = '18/%d calculating percentage data missing for historical information' % total_for_loops):
        x = historical_information.loc[i,'5.schedule_detail']
        y=x.shape
        percentage_null[i]=x.isnull().sum().sum()/(y[0]*y[1])*100

    historical_information['percentage_null'] = pd.Series(percentage_null, index=historical_information.index)

    final_percentage_null_days = (len(historical_information[historical_information.percentage_null > 0])/len(historical_information))*100
    historical_information = historical_information.drop(historical_information[historical_information.percentage_null > 0].index).reset_index(drop = True)

    #sort historical_information by departure order each day
    origin_departure_time_HSP = [None]*len(historical_information)
    k=0
    for i in tqdm(range(0,len(historical_information))):
        x = historical_information.loc[i,'5.schedule_detail']
        origin_departure_time_HSP[k] = x.loc[0,'actual_td']
        k = k+1
    historical_information['origin_departure_time'] = pd.Series(origin_departure_time_HSP, index=historical_information.index)

    historical_information = historical_information.sort_values(by=['1.date','origin_departure_time']).reset_index()

    dates = historical_information.loc[:,'1.date'].to_frame()
    dates_unique = dates.duplicated('1.date', keep='first').tolist()
    dates['duplicate_list'] = pd.Series(dates_unique, index=dates.index)
    dates = dates[~dates.duplicate_list].drop(['duplicate_list'], axis=1).reset_index(drop = True)
    k = 0
    order_day = [None]*len(historical_information)
    for i in tqdm(range(0,len(dates))):
        date = dates.loc[i,'1.date']
        day_schedule = historical_information[historical_information['1.date']==date].reset_index(drop = True)
        for j in range(0,len(day_schedule)):
            index = day_schedule.loc[j,'index']
            order_day[k] = j+1
            k = k+1
            
    historical_information['departure_order'] = pd.Series(order_day, index=historical_information.index)
    return historical_information, station_dwell_time_unique, OD_pairs_unique