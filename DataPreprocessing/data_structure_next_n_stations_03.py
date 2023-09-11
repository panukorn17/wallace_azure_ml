# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:40:48 2018

@author: pt1114
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:59:58 2018

@author: Panukorn
"""
#just for uni computers
#if any new packages are installed dont forget to install it on the harddrive
import sys
sys.path.append('D:/Year 4/Dissertation/anaconda3/lib/python3.6/site-packages')


import pandas as pd
from tqdm import tqdm
import _pickle as cPickle
import datetime
from datetime import timedelta

def data_structure_next_n_stations(historical_info_open, station_dwell_time_unique_open, OD_pairs_unique_open, n_prediction_steps):
    no_all_points = historical_info_open.loc[:,'4.stops'].sum()
    format = '%H%M'    

    #non-featuristic
    #arrival and departure times
    arrival_time = [None]*no_all_points
    arrival_date_time = [None]*no_all_points
    arrival_predicted = [None]*no_all_points
    arrival_predicted_date_time = [None]*no_all_points
    departure_time = [None]*no_all_points
    departure_date_time = [None]*no_all_points
    departure_predicted = [None]*no_all_points
    departure_predicted_date_time = [None]*no_all_points
    origin_departure_time = [None]*no_all_points
    arrival_time_next_n_station = [None]*no_all_points
    arrival_date_time_next_n_station = [None]*no_all_points
    departure_time_next_n_station = [None]*no_all_points
    departure_date_time_next_n_station = [None]*no_all_points
    arrival_time_next_n_station_val_predicted = [None]*no_all_points
    arrival_predicted_date_time_next_n_station = [None]*no_all_points
    departure_time_next_n_station_val_predicted = [None]*no_all_points
    departure_predicted_date_time_next_n_station = [None]*no_all_points

    #featuristic
    #classification 
    date = [None]*no_all_points
    day = [None]*no_all_points
    month = [None]*no_all_points
    RID = [None]*no_all_points
    origin = [None]*no_all_points
    destination = [None]*no_all_points
    prev_station = [None]*no_all_points
    current_station = [None]*no_all_points
    next_n_station = [None]*no_all_points
    next_n_minue_one_station = [None]*no_all_points
    number_stops = [None]*no_all_points
    origin_departure_period = [None]*no_all_points
    delay_reason = [None]*no_all_points
    order_of_journey = [None]*no_all_points
    order_of_departure = [None]*no_all_points
    is_origin = [None]*no_all_points
    is_destination = [None]*no_all_points
    direction = [None]*no_all_points

    #dwell times
    dwell_time = [None]*no_all_points
    dwell_time_predicted = [None]*no_all_points
    total_dwell_time_station = [None]*no_all_points
    total_dwell_time_station_predicted = [None]*no_all_points
    average_dwell_time_station = [None]*no_all_points
    average_dwell_time_station_predicted = [None]*no_all_points
    dwell_time_next_n_station = [None]*no_all_points
    dwell_time_next_n_station_predicted = [None]*no_all_points
    dwell_time_next_n_station_average = [None]*no_all_points

    #travel time in minutes
    travel_time_next_n_station = [None]*no_all_points
    travel_time_next_n_station_predicted = [None]*no_all_points
    travel_time_prev_station = [None]*no_all_points
    travel_time_prev_station_predicted = [None]*no_all_points
    travel_time_next_n_station_average = [None]*no_all_points
    travel_time_next_n_station_predicted_average = [None]*no_all_points

    #prediction value
    deviation_from_arrival = [None]*no_all_points
    deviation_from_arrival_next_n_station = [None]*no_all_points
    deviation_from_departure = [None]*no_all_points
    deviation_from_departure_next_n_station = [None]*no_all_points
    cumulative_departure_delay = [None]*no_all_points
    cumulative_arrival_delay = [None]*no_all_points

    #predefine values to be 0
    cumulative_departure_delay_in = 0
    cumulative_arrival_delay_in = 0
    k=0

    for i in tqdm(range(0,len(historical_info_open))):
        x = historical_info_open.loc[i,'5.schedule_detail']
        departure_predicted_day_add = 0
        departure_day_add = 0
        arrival_day_add = 0
        arrival_predicted_day_add = 0
        for j in range(0,len(x)):
            
            #classification 
            date[k] = historical_info_open.loc[i,'1.date']
            day[k] = datetime.datetime.strptime(date[k], '%Y-%m-%d').weekday()
            month[k] = datetime.datetime.strptime(date[k], '%Y-%m-%d').month
            RID[k] = str(historical_info_open.loc[i,'RID'])
            origin[k] = historical_info_open.loc[i,'2.origin']
            destination[k] = historical_info_open.loc[i,'3.destination']
            current_station[k] = x.loc[j,'location']
            number_stops[k] = historical_info_open.loc[i,'4.stops']
            delay_reason[k] = x.loc[j,'late_canc_reason']
            origin_departure_time[k] = x.loc[0,'actual_td']
            order_of_journey[k] = j+1
            order_of_departure[k] = historical_info_open.loc[i,'departure_order']
            stops_left = number_stops[k]- order_of_journey[k]
            
            if '0600' < origin_departure_time[k] < '0700':
                origin_departure_period[k] = 0
            
            elif '0700' < origin_departure_time[k] < '0800':
                origin_departure_period[k] = 1
            
            elif '0800' < origin_departure_time[k] < '0900':
                origin_departure_period[k] = 2
            
            else:
                origin_departure_period[k] = 3
            
            if historical_info_open.loc[i,'3.destination'] == 'PAD':
                direction[k] = 1
            else:
                direction[k] = 0
            
            #dwell times
            dwell_time[k] = x.loc[j,'dwell_time']
            dwell_time_predicted[k] = x.loc[j,'dwell_time_predicted']
            
            #travel time
            travel_time_prev_station[k] = x.loc[j,'travel_time']
            travel_time_prev_station_predicted[k] = x.loc[j,'travel_time_predicted']
            
            #arrival and departure times
            arrival_time[k] = x.loc[j,'actual_ta']
            arrival_predicted[k] = x.loc[j,'gbtt_pta']
            departure_time[k] = x.loc[j,'actual_td']
            departure_predicted[k] = x.loc[j,'gbtt_ptd']
            
            #arrival and departure date times
            if j==0:
                #predicted departure
                departure_predicted_strptime_1 = datetime.datetime.strptime(departure_predicted[k], '%H%M')
                departure_predicted_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_predicted_day_add)
                departure_predicted_date_time[k] = departure_predicted_strpdatetime_1.replace(hour=departure_predicted_strptime_1.hour, minute=departure_predicted_strptime_1.minute)
                #departure
                departure_strptime_1 = datetime.datetime.strptime(departure_predicted[k], '%H%M')
                departure_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_day_add)
                departure_date_time[k] = departure_strpdatetime_1.replace(hour=departure_strptime_1.hour, minute=departure_strptime_1.minute)
                #predicted arrival
                arrival_predicted_date_time[k] = 'starting'
                #actual arrival
                arrival_date_time[k] = 'starting'
            elif len(x) == 2:
                #predicted departure
                departure_predicted_date_time[k] = 'terminating'
            #departure
                departure_date_time[k] = 'terminating'
                #predicted arrival
                arrival_predicted_strptime_1 = datetime.datetime.strptime(x.loc[j,'gbtt_pta'], '%H%M')
                departure_strptime_0 = datetime.datetime.strptime(x.loc[0,'gbtt_ptd'], '%H%M')
                diff = (arrival_predicted_strptime_1-departure_strptime_0).total_seconds() / 60.0
                if diff<0:
                    arrival_predicted_day_add = arrival_predicted_day_add + 1
                    arrival_date_1 = date[k]
                    arrival_predicted_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_predicted_day_add)
                    arrival_predicted_date_time[k] = arrival_predicted_strpdatetime_1.replace(hour=arrival_predicted_strptime_1.hour, minute=arrival_predicted_strptime_1.minute)
                else:
                    arrival_date_1 = date[k]
                    arrival_predicted_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_predicted_day_add)
                    arrival_predicted_date_time[k] = arrival_predicted_strpdatetime_1.replace(hour=arrival_predicted_strptime_1.hour, minute=arrival_predicted_strptime_1.minute)
                #arrival
                arrival_strptime_1 = datetime.datetime.strptime(arrival_time[k], '%H%M')
                departure_strptime_0 = datetime.datetime.strptime(x.loc[0,'gbtt_ptd'], '%H%M')
                diff = (arrival_strptime_1-departure_strptime_0).total_seconds() / 60.0
                if diff<0:
                    arrival_day_add = arrival_day_add + 1
                    arrival_date_1 = date[k]
                    arrival_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_day_add)
                    arrival_date_time[k] = arrival_strpdatetime_1.replace(hour=arrival_strptime_1.hour, minute=arrival_strptime_1.minute)
                else:
                    arrival_date_1 = date[k]
                    arrival_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_day_add)
                    arrival_date_time[k] = arrival_strpdatetime_1.replace(hour=arrival_strptime_1.hour, minute=arrival_strptime_1.minute)
    
            elif j==1:
                #predicted departure
                departure_predicted_strptime_1 = datetime.datetime.strptime(x.loc[j,'gbtt_ptd'], '%H%M')
                departure_predicted_strptime_2 = datetime.datetime.strptime(x.loc[j-1,'gbtt_ptd'], '%H%M')
                diff = (departure_predicted_strptime_1-departure_predicted_strptime_2).total_seconds() / 60.0
                if diff<0:
                    departure_predicted_day_add = departure_predicted_day_add + 1
                    departure_predicted_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_predicted_day_add)
                    departure_predicted_date_time[k] = departure_predicted_strpdatetime_1.replace(hour=departure_predicted_strptime_1.hour, minute=departure_predicted_strptime_1.minute)
                else:
                    departure_predicted_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_predicted_day_add)
                    departure_predicted_date_time[k] = departure_predicted_strpdatetime_1.replace(hour=departure_predicted_strptime_1.hour, minute=departure_predicted_strptime_1.minute)
                #departure
                departure_strptime_1 = datetime.datetime.strptime(x.loc[j,'actual_td'], '%H%M')
                departure_strptime_2 = datetime.datetime.strptime(x.loc[j-1,'actual_td'], '%H%M')
                diff = (departure_strptime_1-departure_strptime_2).total_seconds() / 60.0
                if diff<0:
                    departure_day_add = departure_day_add + 1
                    departure_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_day_add)
                    departure_date_time[k] = departure_strpdatetime_1.replace(hour=departure_strptime_1.hour, minute=departure_strptime_1.minute)
                else:
                    departure_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_day_add)
                    departure_date_time[k] = departure_strpdatetime_1.replace(hour=departure_strptime_1.hour, minute=departure_strptime_1.minute)
                #predicted arrival
                arrival_predicted_strptime_1 = datetime.datetime.strptime(x.loc[j,'gbtt_pta'], '%H%M')
                departure_strptime_0 = datetime.datetime.strptime(x.loc[0,'gbtt_ptd'], '%H%M')
                diff = (arrival_predicted_strptime_1-departure_strptime_0).total_seconds() / 60.0
                if diff<0:
                    arrival_predicted_day_add = arrival_predicted_day_add + 1
                    arrival_date_1 = date[k]
                    arrival_predicted_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_predicted_day_add)
                    arrival_predicted_date_time[k] = arrival_predicted_strpdatetime_1.replace(hour=arrival_predicted_strptime_1.hour, minute=arrival_predicted_strptime_1.minute)
                else:
                    arrival_date_1 = date[k]
                    arrival_predicted_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_predicted_day_add)
                    arrival_predicted_date_time[k] = arrival_predicted_strpdatetime_1.replace(hour=arrival_predicted_strptime_1.hour, minute=arrival_predicted_strptime_1.minute)
                #arrival
                arrival_strptime_1 = datetime.datetime.strptime(arrival_time[k], '%H%M')
                departure_strptime_0 = datetime.datetime.strptime(x.loc[0,'gbtt_ptd'], '%H%M')
                diff = (arrival_strptime_1-departure_strptime_0).total_seconds() / 60.0
                if diff<0:
                    arrival_day_add = arrival_day_add + 1
                    arrival_date_1 = date[k]
                    arrival_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_day_add)
                    arrival_date_time[k] = arrival_strpdatetime_1.replace(hour=arrival_strptime_1.hour, minute=arrival_strptime_1.minute)
                else:
                    arrival_date_1 = date[k]
                    arrival_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_day_add)
                    arrival_date_time[k] = arrival_strpdatetime_1.replace(hour=arrival_strptime_1.hour, minute=arrival_strptime_1.minute)
            elif j==len(x)-1:
                #predicted departure
                departure_predicted_date_time[k] = 'terminating'
                #departure
                departure_date_time[k] = 'terminating'
                #predicted arrival
                arrival_predicted_strptime_1 = datetime.datetime.strptime(x.loc[j,'gbtt_pta'], '%H%M')
                arrival_predicted_strptime_2 = datetime.datetime.strptime(x.loc[j-1,'gbtt_pta'], '%H%M')
                diff = (arrival_predicted_strptime_1-arrival_predicted_strptime_2).total_seconds() / 60.0
                if diff<0:
                    arrival_predicted_day_add = arrival_predicted_day_add + 1
                    arrival_date_1 = date[k]
                    arrival_predicted_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_predicted_day_add)
                    arrival_predicted_date_time[k] = arrival_predicted_strpdatetime_1.replace(hour=arrival_predicted_strptime_1.hour, minute=arrival_predicted_strptime_1.minute)
                else:
                    arrival_date_1 = date[k]
                    arrival_predicted_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_predicted_day_add)
                    arrival_predicted_date_time[k] = arrival_predicted_strpdatetime_1.replace(hour=arrival_predicted_strptime_1.hour, minute=arrival_predicted_strptime_1.minute)
                #arrival
                arrival_strptime_1 = datetime.datetime.strptime(x.loc[j,'actual_ta'], '%H%M')
                arrival_strptime_2 = datetime.datetime.strptime(x.loc[j-1,'actual_ta'], '%H%M')
                diff = (arrival_strptime_1-arrival_strptime_2).total_seconds() / 60.0
                if diff<0:
                    arrival_day_add = arrival_day_add + 1
                    arrival_date_1 = date[k]
                    arrival_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_day_add)
                    arrival_date_time[k] = arrival_strpdatetime_1.replace(hour=arrival_strptime_1.hour, minute=arrival_strptime_1.minute)
                else:
                    arrival_date_1 = date[k]
                    arrival_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_day_add)
                    arrival_date_time[k] = arrival_strpdatetime_1.replace(hour=arrival_strptime_1.hour, minute=arrival_strptime_1.minute)
                
            else:
                #predicted departure
                departure_predicted_strptime_1 = datetime.datetime.strptime(x.loc[j,'gbtt_ptd'], '%H%M')
                departure_predicted_strptime_2 = datetime.datetime.strptime(x.loc[j-1,'gbtt_ptd'], '%H%M')
                diff = (departure_predicted_strptime_1-departure_predicted_strptime_2).total_seconds() / 60.0
                if diff<0:
                    departure_predicted_day_add = departure_predicted_day_add + 1
                    departure_predicted_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_predicted_day_add)
                    departure_predicted_date_time[k] = departure_predicted_strpdatetime_1.replace(hour=departure_predicted_strptime_1.hour, minute=departure_predicted_strptime_1.minute)
                else:
                    departure_predicted_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_predicted_day_add)
                    departure_predicted_date_time[k] = departure_predicted_strpdatetime_1.replace(hour=departure_predicted_strptime_1.hour, minute=departure_predicted_strptime_1.minute)
                #departure
                departure_strptime_1 = datetime.datetime.strptime(x.loc[j,'actual_td'], '%H%M')
                departure_strptime_2 = datetime.datetime.strptime(x.loc[j-1,'actual_td'], '%H%M')
                diff = (departure_strptime_1-departure_strptime_2).total_seconds() / 60.0
                if diff<0:
                    departure_day_add = departure_day_add + 1
                    departure_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_day_add)
                    departure_date_time[k] = departure_strpdatetime_1.replace(hour=departure_strptime_1.hour, minute=departure_strptime_1.minute)
                else:
                    departure_strpdatetime_1 = datetime.datetime.strptime(date[k], '%Y-%m-%d') + timedelta(days=departure_day_add)
                    departure_date_time[k] = departure_strpdatetime_1.replace(hour=departure_strptime_1.hour, minute=departure_strptime_1.minute)
                #predicted arrival
                arrival_predicted_strptime_1 = datetime.datetime.strptime(x.loc[j,'gbtt_pta'], '%H%M')
                arrival_predicted_strptime_2 = datetime.datetime.strptime(x.loc[j-1,'gbtt_pta'], '%H%M')
                diff = (arrival_predicted_strptime_1-arrival_predicted_strptime_2).total_seconds() / 60.0
                if diff<0:
                    arrival_predicted_day_add = arrival_predicted_day_add + 1
                    arrival_date_1 = date[k]
                    arrival_predicted_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_predicted_day_add)
                    arrival_predicted_date_time[k] = arrival_predicted_strpdatetime_1.replace(hour=arrival_predicted_strptime_1.hour, minute=arrival_predicted_strptime_1.minute)
                else:
                    arrival_date_1 = date[k]
                    arrival_predicted_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_predicted_day_add)
                    arrival_predicted_date_time[k] = arrival_predicted_strpdatetime_1.replace(hour=arrival_predicted_strptime_1.hour, minute=arrival_predicted_strptime_1.minute)
                #arrival
                arrival_strptime_1 = datetime.datetime.strptime(x.loc[j,'actual_ta'], '%H%M')
                arrival_strptime_2 = datetime.datetime.strptime(x.loc[j-1,'actual_ta'], '%H%M')
                diff = (arrival_strptime_1-arrival_strptime_2).total_seconds() / 60.0
                if diff<0:
                    arrival_day_add = arrival_day_add + 1
                    arrival_date_1 = date[k]
                    arrival_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_day_add)
                    arrival_date_time[k] = arrival_strpdatetime_1.replace(hour=arrival_strptime_1.hour, minute=arrival_strptime_1.minute)
                else:
                    arrival_date_1 = date[k]
                    arrival_strpdatetime_1 = datetime.datetime.strptime(arrival_date_1, '%Y-%m-%d') + timedelta(days=arrival_day_add)
                    arrival_date_time[k] = arrival_strpdatetime_1.replace(hour=arrival_strptime_1.hour, minute=arrival_strptime_1.minute)
                    
            if stops_left < n_prediction_steps:
                
                next_n_station[k] = 'terminating'
                next_n_minue_one_station[k] = 'terminating'
                arrival_time_next_n_station[k] = 'terminating'
                departure_time_next_n_station[k] = 'terminating'
                arrival_time_next_n_station_val_predicted[k] = 'terminating'
                departure_time_next_n_station_val_predicted[k] = 'terminating'
                dwell_time_next_n_station[k] = 0
                dwell_time_next_n_station_predicted[k] = 0
                dwell_time_next_n_station_average[k] = 0
                travel_time_next_n_station[k] = 0
                travel_time_next_n_station_predicted[k] = 0
                travel_time_next_n_station_average[k] = 0
                travel_time_next_n_station_predicted_average[k] = 0
                deviation_from_arrival_next_n_station[k] = 0
                deviation_from_departure_next_n_station[k] = 0
                
                index = station_dwell_time_unique_open['1.station'].tolist().index(x.loc[j,'location'])
                total_dwell_time_station[k] = station_dwell_time_unique_open.loc[index,'2.total_dwell_time']
                total_dwell_time_station_predicted[k] = station_dwell_time_unique_open.loc[index,'3.total_dwell_time_predicted']
                average_dwell_time_station[k] = station_dwell_time_unique_open.loc[index,'4.average_dwell_time']
                average_dwell_time_station_predicted[k] = station_dwell_time_unique_open.loc[index,'5.average_dwell_time_predicted']
                
                #classification 
                if j == 0 :
                    prev_station[k] = 'starting'
                    is_origin[k] = 1
                else:
                    prev_station[k] = x.loc[j-1,'location']
                    is_origin[k] = 0
                    
                if j == len(x) - 1 :
                    is_destination[k] = 1
                else:
                    is_destination[k] = 0
                
                #prediction value
                arrival_time_val = arrival_time[k]
                arrival_predicted_val = arrival_predicted[k]
                
                departure_time_val = departure_time[k]
                departure_predicted_val = departure_predicted[k]
                
                if arrival_time_val == 'starting':
                    deviation_from_arrival[k] = 0
                    
                else:
                    arrival_time_val = datetime.datetime.strptime(arrival_time_val, format)
                    arrival_predicted_val = datetime.datetime.strptime(arrival_predicted_val, format)
                    deviation_from_arrival[k] = (arrival_time_val - arrival_predicted_val).total_seconds() / 60.0
                    
                    if deviation_from_arrival[k]>500: #meaning that train actaully arrives before midnight
                        #arrival_predicted_val = datetime.datetime.strptime(arrival_predicted_val, format) + timedelta(days=1)
                        arrival_predicted_val = arrival_predicted_val + timedelta(days=1)
                        deviation_from_arrival[k] = (arrival_time_val - arrival_predicted_val).total_seconds() / 60.0
                        
                    if deviation_from_arrival[k]<-500: #meaning that train actaully arrives after midnight
                        #arrival_time_val = datetime.datetime.strptime(arrival_time_val, format) + timedelta(days=1)
                        deviation_from_arrival[k] = (arrival_time_val - arrival_predicted_val).total_seconds() / 60.0
                
                if departure_time_val == 'terminating':
                    deviation_from_departure[k] = 0
                    
                elif j == len(x)-(n_prediction_steps+1):
                    
                    departure_time_val = datetime.datetime.strptime(departure_time_val, format)
                    departure_predicted_val = datetime.datetime.strptime(departure_predicted_val, format)
                    deviation_from_departure[k] = (departure_time_val - departure_predicted_val).total_seconds() / 60.0
                    
                else:
                    departure_time_val = datetime.datetime.strptime(departure_time_val, format)
                    departure_predicted_val = datetime.datetime.strptime(departure_predicted_val, format)
                    deviation_from_departure[k] = (departure_time_val - departure_predicted_val).total_seconds() / 60.0
                    
                    
                    if deviation_from_departure[k]>500: #meaning that train actaully departs before midnight
                        departure_predicted_val = datetime.datetime.strptime(departure_predicted_val, format) + timedelta(days=1)
                        deviation_from_departure[k] = (departure_time_val - departure_predicted_val).total_seconds() / 60.0
                        
                    if deviation_from_departure[k]<-500: #meaning that train actaully departs after midnight
                        departure_time_val = datetime.datetime.strptime(departure_time_val, format) + timedelta(days=1)
                        deviation_from_departure[k] = (departure_time_val - departure_predicted_val).total_seconds() / 60.0
                
                if j == 0:
                    cumulative_departure_delay[k] = cumulative_departure_delay_in + deviation_from_departure[k] 
                    cumulative_arrival_delay[k] = cumulative_departure_delay_in + deviation_from_arrival[k]
                    
                else:
                    cumulative_departure_delay[k] = cumulative_departure_delay[k-1] + deviation_from_departure[k] 
                    cumulative_arrival_delay[k] = cumulative_arrival_delay[k-1] + deviation_from_arrival[k]
                
            else:
                
                if j == len(x) - 1:     #i.e. if this is the final station of the journey
                    dwell_time_next_n_station_predicted[k] = 0
                    dwell_time_next_n_station_average[k] = 0
                    dwell_time_next_n_station[k] = 0
                else: 
                    dwell_time_next_n_station[k] = x.loc[j+1,'dwell_time']
                    dwell_time_next_n_station_predicted[k] = x.loc[j+1,'dwell_time_predicted']
                    index_next_station = station_dwell_time_unique_open['1.station'].tolist().index(x.loc[j+1,'location'])
                    dwell_time_next_n_station_average[k] = station_dwell_time_unique_open.loc[index_next_station,'4.average_dwell_time']
                    
                index = station_dwell_time_unique_open['1.station'].tolist().index(x.loc[j,'location'])
                total_dwell_time_station[k] = station_dwell_time_unique_open.loc[index,'2.total_dwell_time']
                total_dwell_time_station_predicted[k] = station_dwell_time_unique_open.loc[index,'3.total_dwell_time_predicted']
                average_dwell_time_station[k] = station_dwell_time_unique_open.loc[index,'4.average_dwell_time']
                average_dwell_time_station_predicted[k] = station_dwell_time_unique_open.loc[index,'5.average_dwell_time_predicted']
                
            
                #classification 
                if j == 0 :
                    prev_station[k] = 'starting'
                    is_origin[k] = 1
                else:
                    prev_station[k] = x.loc[j-1,'location']
                    is_origin[k] = 0
                
                if j == len(x)-1:
                    next_n_station[k] = 'terminating'
                    next_n_minue_one_station[k] = x.loc[j+n_prediction_steps-1,'location']
                    travel_time_next_n_station[k] = 0
                    travel_time_next_n_station_predicted[k] = 0
                    is_destination[k] = 1
                    arrival_time_next_n_station[k] = 'terminating'
                    departure_time_next_n_station[k] = 'terminating'
                    arrival_time_next_n_station_val_predicted[k] = 'terminating'
                    departure_time_next_n_station_val_predicted[k] = 'terminating'
                    travel_time_next_n_station_average[k] = 0
                    travel_time_next_n_station_predicted_average[k] = 0
                
                elif j == len(x)- (n_prediction_steps+1):
                    next_n_station[k] = x.loc[j+n_prediction_steps,'location']
                    next_n_minue_one_station[k] = x.loc[j+n_prediction_steps-1,'location']
                    travel_time_next_n_station[k] = x.loc[j+n_prediction_steps,'travel_time']
                    travel_time_next_n_station_predicted[k] = x.loc[j+n_prediction_steps,'travel_time_predicted']
                    is_destination[k] = 0
                    arrival_time_next_n_station[k] = x.loc[j+n_prediction_steps,'actual_ta']
                    departure_time_next_n_station[k] = x.loc[j+n_prediction_steps,'actual_td']
                    arrival_time_next_n_station_val_predicted[k] = x.loc[j+n_prediction_steps,'gbtt_pta']
                    departure_time_next_n_station_val_predicted[k] = 'terminating'
                    OD_data = OD_pairs_unique_open[(OD_pairs_unique_open['1.origin']==next_n_minue_one_station[k])&(OD_pairs_unique_open['2.destination']==next_n_station[k])].reset_index(drop = True)
                    travel_time_next_n_station_average[k] = OD_data.loc[0,'average_travel_time']
                    travel_time_next_n_station_predicted_average[k] = OD_data.loc[0,'average_travel_time_predicted']
                    
                else: 
                    next_n_station[k] = x.loc[j+n_prediction_steps,'location']
                    next_n_minue_one_station[k] = x.loc[j+n_prediction_steps-1,'location']
                    travel_time_next_n_station[k] = x.loc[j+n_prediction_steps,'travel_time']
                    travel_time_next_n_station_predicted[k] = x.loc[j+n_prediction_steps,'travel_time_predicted']
                    is_destination[k] = 0
                    arrival_time_next_n_station[k] = x.loc[j+n_prediction_steps,'actual_ta']
                    departure_time_next_n_station[k] = x.loc[j+n_prediction_steps,'actual_td']
                    arrival_time_next_n_station_val_predicted[k] = x.loc[j+n_prediction_steps,'gbtt_pta']
                    departure_time_next_n_station_val_predicted[k] = x.loc[j+n_prediction_steps,'gbtt_ptd']
                    OD_data = OD_pairs_unique_open[(OD_pairs_unique_open['1.origin']==next_n_minue_one_station[k])&(OD_pairs_unique_open['2.destination']==next_n_station[k])].reset_index(drop = True)
                    travel_time_next_n_station_average[k] = OD_data.loc[0,'average_travel_time']
                    travel_time_next_n_station_predicted_average[k] = OD_data.loc[0,'average_travel_time_predicted']
                    
                    
                #prediction value
                arrival_time_val = arrival_time[k]
                arrival_predicted_val = arrival_predicted[k]
                arrival_next_n_station_val = arrival_time_next_n_station[k]
                arrival_next_n_station_val_predicted = arrival_time_next_n_station_val_predicted[k]
                
                departure_time_val = departure_time[k]
                departure_predicted_val = departure_predicted[k]
                departure_next_n_station_val = departure_time_next_n_station[k]
                departure_next_n_station_val_predicted = departure_time_next_n_station_val_predicted[k]
                
                if arrival_time_val == 'starting':
                    deviation_from_arrival[k] = 0
                    
                else:
                    arrival_time_val = datetime.datetime.strptime(arrival_time_val, format)
                    arrival_predicted_val = datetime.datetime.strptime(arrival_predicted_val, format)
                    deviation_from_arrival[k] = (arrival_time_val - arrival_predicted_val).total_seconds() / 60.0
                    
                    if deviation_from_arrival[k]>500: #meaning that train actaully arrives before midnight
                        #arrival_predicted_val = datetime.datetime.strptime(arrival_predicted_val, format) + timedelta(days=1)
                        arrival_predicted_val = arrival_predicted_val + timedelta(days=1)
                        deviation_from_arrival[k] = (arrival_time_val - arrival_predicted_val).total_seconds() / 60.0
                        
                    if deviation_from_arrival[k]<-500: #meaning that train actaully arrives after midnight
                        #arrival_time_val = datetime.datetime.strptime(arrival_time_val, format) + timedelta(days=1)
                        arrival_time_val = arrival_time_val + timedelta(days=1)
                        deviation_from_arrival[k] = (arrival_time_val - arrival_predicted_val).total_seconds() / 60.0
                
                if departure_time_val == 'terminating':
                    deviation_from_departure[k] = 0
                    deviation_from_arrival_next_n_station[k]  = 0
                    deviation_from_departure_next_n_station[k] = 0
                    
                elif j == len(x)-(n_prediction_steps+1):
                    deviation_from_departure_next_n_station[k] = 0
                    
                    departure_time_val = datetime.datetime.strptime(departure_time_val, format)
                    departure_predicted_val = datetime.datetime.strptime(departure_predicted_val, format)
                    deviation_from_departure[k] = (departure_time_val - departure_predicted_val).total_seconds() / 60.0
                    
                    arrival_next_n_station_val = datetime.datetime.strptime(arrival_next_n_station_val, format)
                    arrival_next_n_station_val_predicted = datetime.datetime.strptime(arrival_next_n_station_val_predicted, format)
                    deviation_from_arrival_next_n_station[k] = (arrival_next_n_station_val - arrival_next_n_station_val_predicted).total_seconds() / 60.0
                    
                else:
                    departure_time_val = datetime.datetime.strptime(departure_time_val, format)
                    departure_predicted_val = datetime.datetime.strptime(departure_predicted_val, format)
                    deviation_from_departure[k] = (departure_time_val - departure_predicted_val).total_seconds() / 60.0
                    
                    departure_next_n_station_val = datetime.datetime.strptime(departure_next_n_station_val, format)
                    departure_next_n_station_val_predicted = datetime.datetime.strptime(departure_next_n_station_val_predicted, format)
                    deviation_from_departure_next_n_station[k] = (departure_next_n_station_val - departure_next_n_station_val_predicted).total_seconds() / 60.0
                    
                    arrival_next_n_station_val = datetime.datetime.strptime(arrival_next_n_station_val, format)
                    arrival_next_n_station_val_predicted = datetime.datetime.strptime(arrival_next_n_station_val_predicted, format)
                    deviation_from_arrival_next_n_station[k] = (arrival_next_n_station_val - arrival_next_n_station_val_predicted).total_seconds() / 60.0
        
                    
                    if deviation_from_departure[k]>500: #meaning that train actaully departs before midnight
                        #departure_predicted_val = datetime.datetime.strptime(departure_predicted_val, format) + timedelta(days=1)
                        departure_predicted_val = departure_predicted_val + timedelta(days=1)
                        deviation_from_departure[k] = (departure_time_val - departure_predicted_val).total_seconds() / 60.0
                        
                    if deviation_from_departure[k]<-500: #meaning that train actaully departs after midnight
                        #departure_time_val = datetime.datetime.strptime(departure_time_val, format) + timedelta(days=1)
                        departure_time_val = departure_time_val + timedelta(days=1)
                        deviation_from_departure[k] = (departure_time_val - departure_predicted_val).total_seconds() / 60.0
                
                if j == 0:
                    cumulative_departure_delay[k] = cumulative_departure_delay_in + deviation_from_departure[k] 
                    cumulative_arrival_delay[k] = cumulative_departure_delay_in + deviation_from_arrival[k]
                    
                else:
                    cumulative_departure_delay[k] = cumulative_departure_delay[k-1] + deviation_from_departure[k] 
                    cumulative_arrival_delay[k] = cumulative_arrival_delay[k-1] + deviation_from_arrival[k]
        
            k = k+1
            
        cumulative_departure_delay_in = 0
        cumulative_arrival_delay_in = 0        
            

    DL_data_next_station = pd.DataFrame({
                            'arrival_time':arrival_time,
                            'arrival_date_time':arrival_date_time,
                            'arrival_predicted':arrival_predicted,
                            'arrival_predicted_date_time':arrival_predicted_date_time,
                            'departure_time':departure_time,
                            'departure_date_time':departure_date_time,
                            'departure_predicted':departure_predicted,
                            'departure_predicted_date_time':departure_predicted_date_time,
                            'origin_departure_time':origin_departure_time,
                            'arrival_time_next_'+str(n_prediction_steps)+'_station': arrival_time_next_n_station,
                            'departure_time_next_'+str(n_prediction_steps)+'_station': departure_time_next_n_station,
                            'arrival_time_next_'+str(n_prediction_steps)+'_station_val_predicted': arrival_time_next_n_station_val_predicted,
                            'departure_time_next_'+str(n_prediction_steps)+'_station_val_predicted': departure_time_next_n_station_val_predicted,
                            
                            'date':date,
                            'day':day,
                            'month':month,
                            'RID':RID,
                            'origin':origin,
                            'destination':destination,
                            'previous_station':prev_station,
                            'current_station':current_station,
                            'next_'+str(n_prediction_steps)+'_station':next_n_station,
                            'next_n_minue_one_station':next_n_minue_one_station,
                            'number_stops':number_stops,
                            'origin_departure_period':origin_departure_period,
                            'delay_reason':delay_reason,
                            'is_origin':is_origin,
                            'is_destination':is_destination,
                            'order_of_journey':order_of_journey,
                            'order_of_departure':order_of_departure,
                            'direction':direction,
                            
                            'dwell_time_curr_station':dwell_time,
                            'dwell_time_curr_station_predicted':dwell_time_predicted,
                            'dwell_time_next_'+str(n_prediction_steps)+'_station':dwell_time_next_n_station,
                            'dwell_time_next_'+str(n_prediction_steps)+'_station_predicted':dwell_time_next_n_station_predicted,
                            'dwell_time_next_'+str(n_prediction_steps)+'_station_average':dwell_time_next_n_station_average,
                            'dwell_time_total_curr_station':total_dwell_time_station,
                            'dwell_time_total_curr_station_predicted':total_dwell_time_station_predicted,
                            'dwell_time_total_curr_station_average':average_dwell_time_station,
                            'dwell_time_total_curr_station_average_predicted':average_dwell_time_station_predicted,
                            
                            'travel_time_next_'+str(n_prediction_steps)+'_station':travel_time_next_n_station,
                            'travel_time_next_'+str(n_prediction_steps)+'_station_predicted':travel_time_next_n_station_predicted,
                            'travel_time_prev_station':travel_time_prev_station,
                            'travel_time_prev_station_predicted':travel_time_prev_station_predicted,   
                            'travel_time_next_'+str(n_prediction_steps)+'_station_average':travel_time_next_n_station_average,
                            'travel_time_next_'+str(n_prediction_steps)+'_station_predicted_average':travel_time_next_n_station_predicted_average,
                            
                            'deviation_from_arrival':deviation_from_arrival, 
                            'deviation_from_departure':deviation_from_departure,
                            'deviation_from_arrival_next_'+str(n_prediction_steps)+'_station': deviation_from_arrival_next_n_station,
                            'deviation_from_departure_next_'+str(n_prediction_steps)+'_station':deviation_from_departure_next_n_station,
                            'cumulative_departure_delay':cumulative_departure_delay,
                            'cumulative_arrival_delay':cumulative_arrival_delay                                           
                            })

    k=0
    RID_unique = DL_data_next_station['RID'].unique().tolist()
    for i in tqdm(range(0,len(RID_unique))):
        x = DL_data_next_station.loc[DL_data_next_station['RID']==RID_unique[i]]
        for j in range(0,len(x)):
            #classification
            index = x.index[j]
            number_stops = x.loc[index,'number_stops']
            order_of_journey = x.loc[index,'order_of_journey']
            stops_left = number_stops - order_of_journey
            if stops_left < n_prediction_steps:
                arrival_date_time_next_n_station[k] = 'terminating'
                arrival_predicted_date_time_next_n_station[k] = 'terminating'
                departure_date_time_next_n_station[k] = 'terminating'
                departure_predicted_date_time_next_n_station[k] = 'terminating'
            else:
                arrival_date_time_next_n_station[k] = x.loc[index + n_prediction_steps,'arrival_date_time']
                arrival_predicted_date_time_next_n_station[k] = x.loc[index + n_prediction_steps,'arrival_predicted_date_time']
                departure_date_time_next_n_station[k] =  x.loc[index + n_prediction_steps,'departure_date_time']
                departure_predicted_date_time_next_n_station[k] =  x.loc[index + n_prediction_steps,'departure_predicted_date_time']
            k=k+1
                
    DL_data_next_station.insert(2, 'arrival_date_time_next_n_station', arrival_date_time_next_n_station)
    DL_data_next_station.insert(5, 'arrival_predicted_date_time_next_n_station', arrival_predicted_date_time_next_n_station)
    DL_data_next_station.insert(8, 'departure_date_time_next_n_station', departure_date_time_next_n_station)
    DL_data_next_station.insert(11, 'departure_predicted_date_time_next_n_station', departure_predicted_date_time_next_n_station)

    return DL_data_next_station



            
        