# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:40:43 2020

@author: Teddy Taleongpong
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
#year = '2017'

#n_prediction_steps = 1

#pickle_in = open('D:/Wallace/Wallace-Github/data_preprocess/'+year+' Data/data_next_'+str(n_prediction_steps)+'_station_'+year+'.pickle',"rb")
#dataset = cPickle.load(pickle_in)
def delay_mechanism(dataset):
    #insert new columns
    dataset.insert(loc=48, column='delay_type_2', value = 0)
    #dataset["delay_type_2_desc"] = ""
    dataset.insert(loc=49, column='delay_type_2_desc', value = 0)

    dataset.insert(loc=50, column='delay_type_3', value = 0)
    #dataset["delay_type_3_desc"] = ""
    dataset.insert(loc=51, column='delay_type_3_desc', value = 0)

    dataset.insert(loc=52, column='delay_type_4', value = 0)
    #dataset["delay_type_4_desc"] = ""
    dataset.insert(loc=53, column='delay_type_4_desc', value = 0)

    dataset.insert(loc=54, column='delay_type_5', value = 0)
    #dataset["delay_type_5_desc"] = ""
    dataset.insert(loc=55, column='delay_type_5_desc', value = 0)

    #delay type 1 - self propagation 

    #delay type 2 - arrival backward propagation
    for i in tqdm(range(0,len(dataset))):
        #for a journey
        prop_2_journey = dataset.iloc[i]
        if prop_2_journey["deviation_from_departure"] == 0:
            continue
        #if journey delay in departure
        prop_2_journey
        #filter all journeys on the same day
        prop_2_journey_date = dataset[dataset['date']==prop_2_journey['date']]
        
        prop_2_journey_date
        #filter trains that are travelling to station that current journey is leaving from
        prop_2_journey_date_station = prop_2_journey_date[prop_2_journey_date['next_1_station']==prop_2_journey['current_station']]
        prop_2_journey_date_station
        #filter trains that are scheduled to arrive at current station after journey train is scheduled to leave 
        prop_2_journey_date_station_time = prop_2_journey_date_station[prop_2_journey_date_station['arrival_time_next_1_station_val_predicted']>prop_2_journey['departure_predicted']]
        prop_2_journey_date_station_time
        #filter out trains that is scheduled to arrive after journey train leaves current station
        prop_2_journey_date_station_time_depart = prop_2_journey_date_station_time[prop_2_journey_date_station_time['arrival_time_next_1_station_val_predicted']<prop_2_journey['departure_time']]
        prop_2_journey_date_station_time_depart
        if len(prop_2_journey_date_station_time_depart) == 0:
            continue
            #get delay data
        rid = prop_2_journey["RID"]
        journey_order = prop_2_journey["order_of_journey"]
        delay = prop_2_journey["deviation_from_departure"]
        
        #for loop to go through all filtered journeys
        for j in range(0,len(prop_2_journey_date_station_time_depart)):
            index = prop_2_journey_date_station_time_depart.index[j]
            dataset.loc[index,"delay_type_2"]=dataset.loc[index,"delay_type_2"]+delay
            #delay dict
            delay_list = [rid,journey_order,delay]
            
            #if this cell is 0
            if dataset["delay_type_2_desc"][index] == 0:
                dataset["delay_type_2_desc"][index] = delay_list
            else:
                #if this sell is not 0
                dataset["delay_type_2_desc"][index].extend(delay_list)

            
        


    #delay type 3 - arrival forward propagation
    for i in tqdm(range(0,len(dataset))):
        #for a journey
        prop_3_journey = dataset.iloc[i]
        if prop_3_journey["deviation_from_arrival_next_1_station"] == 0:
            continue
        #if journey delay in departure
        prop_3_journey
        #filter all trains on the same day
        prop_3_journey_date = dataset[dataset['date']==prop_3_journey['date']]
        
        prop_3_journey_date
        #filter  trains that are travelling to station that current journey is travelling to
        prop_3_journey_date_station = prop_3_journey_date[prop_3_journey_date['next_1_station']==prop_3_journey['next_1_station']]
        prop_3_journey_date_station
        #filter trains that are scheduled to arrive at next station after journey train is scheduled to arrive at that station 
        prop_3_journey_date_station_time = prop_3_journey_date_station[prop_3_journey_date_station['arrival_time_next_1_station_val_predicted']>prop_3_journey['arrival_time_next_1_station_val_predicted']]
        prop_3_journey_date_station_time
        #filter out trains that is scheduled to arrive after journey train arrives at that station
        prop_3_journey_date_station_time_arrive = prop_3_journey_date_station_time[prop_3_journey_date_station_time['arrival_time_next_1_station_val_predicted']<prop_3_journey['arrival_time_next_1_station']]
        prop_3_journey_date_station_time_arrive
        if len(prop_3_journey_date_station_time_arrive) == 0:
            continue
            #get delay data
        rid = prop_3_journey["RID"]
        journey_order = prop_3_journey["order_of_journey"]
        delay = prop_3_journey["deviation_from_arrival_next_1_station"]
        
        #for loop to go through all filtered journeys
        for j in range(0,len(prop_3_journey_date_station_time_arrive)):
            index = prop_3_journey_date_station_time_arrive.index[j]
            dataset.loc[index,"delay_type_3"]=dataset.loc[index,"delay_type_3"]+delay
            #delay dict
            delay_list = [rid,journey_order,delay]
            
            #if this cell is 0
            if dataset["delay_type_3_desc"][index] == 0:
                dataset["delay_type_3_desc"][index] = delay_list
            else:
                #if this sell is not 0
                dataset["delay_type_3_desc"][index].extend(delay_list)

            
    #delay type 4 - departure backward propagation
    for i in tqdm(range(0,len(dataset))):
        #for a journey
        prop_4_journey = dataset.iloc[i]
        if prop_4_journey["deviation_from_departure"] == 0:
            continue
        #if journey delay in departure
        prop_4_journey
        #filter all trains on the same day
        prop_4_journey_date = dataset[dataset['date']==prop_4_journey['date']]
        
        prop_4_journey_date
        #filter trains that are departing from the station that current journey is departing from
        prop_4_journey_date_station = prop_4_journey_date[prop_4_journey_date['current_station']==prop_4_journey['current_station']]
        prop_4_journey_date_station
        #filter trains that are scheduled to depart from current station after journey train is scheduled to depart from current station 
        prop_4_journey_date_station_time = prop_4_journey_date_station[prop_4_journey_date_station['departure_predicted']>prop_4_journey['departure_predicted']]
        prop_4_journey_date_station_time
        #filter out trains that is scheduled to depart before journey train departs current station
        prop_4_journey_date_station_time_arrive = prop_4_journey_date_station_time[prop_4_journey_date_station_time['departure_predicted']<prop_4_journey['departure_time']]
        prop_4_journey_date_station_time_arrive
        if len(prop_4_journey_date_station_time_arrive) == 0:
            continue
            #get delay data
        rid = prop_4_journey["RID"]
        journey_order = prop_4_journey["order_of_journey"]
        delay = prop_4_journey["deviation_from_departure"]
        
        #for loop to go through all filtered journeys
        for j in range(0,len(prop_4_journey_date_station_time_arrive)):
            index = prop_4_journey_date_station_time_arrive.index[j]
            dataset.loc[index,"delay_type_4"]=dataset.loc[index,"delay_type_4"]+delay
            #delay dict
            delay_list = [rid,journey_order,delay]
            
            #if this cell is 0
            if dataset["delay_type_4_desc"][index] == 0:
                dataset["delay_type_4_desc"][index] = delay_list
            else:
                #if this sell is not 0
                dataset["delay_type_4_desc"][index].extend(delay_list)

    #delay type 5 - departure forward propagation
    for i in tqdm(range(0,len(dataset))):
        #for a journey
        prop_5_journey = dataset.iloc[i]
        if prop_5_journey["deviation_from_arrival_next_1_station"] == 0:
            continue
        #if journey delay in arrival
        prop_5_journey
        #filter all trains on the same day
        prop_5_journey_date = dataset[dataset['date']==prop_5_journey['date']]
        
        prop_5_journey_date
        #filter trains that are departing from the station that current journey is arrive at
        prop_5_journey_date_station = prop_5_journey_date[prop_5_journey_date['current_station']==prop_5_journey['next_1_station']]
        prop_5_journey_date_station
        #filter trains that are scheduled to depart from current station after journey train is scheduled to arrive at current station 
        prop_5_journey_date_station_time = prop_5_journey_date_station[prop_5_journey_date_station['departure_predicted']>prop_5_journey['arrival_time_next_1_station_val_predicted']]
        prop_5_journey_date_station_time
        #filter out trains that is scheduled to depart before journey train arives at current station
        prop_5_journey_date_station_time_arrive = prop_5_journey_date_station_time[prop_5_journey_date_station_time['departure_predicted']<prop_5_journey['arrival_time_next_1_station']]
        prop_5_journey_date_station_time_arrive
        if len(prop_5_journey_date_station_time_arrive) == 0:
            continue
            #get delay data
        rid = prop_5_journey["RID"]
        journey_order = prop_5_journey["order_of_journey"]
        delay = prop_5_journey["deviation_from_arrival_next_1_station"]
        
        #for loop to go through all filtered journeys
        for j in range(0,len(prop_5_journey_date_station_time_arrive)):
            index = prop_5_journey_date_station_time_arrive.index[j]
            dataset.loc[index,"delay_type_5"]=dataset.loc[index,"delay_type_5"]+delay
            #delay dict
            delay_list = [rid,journey_order,delay]
            
            #if this cell is 0
            if dataset["delay_type_5_desc"][index] == 0:
                dataset["delay_type_5_desc"][index] = delay_list
            else:
                #if this sell is not 0
                dataset["delay_type_5_desc"][index].extend(delay_list)
        

    #test = dataset[(dataset['RID']=="201601041357177") & (dataset['order_of_journey']==1)]

    '''    import _pickle as cPickle
        pickle_out = open('D:/Wallace/Wallace-Github/data_preprocess/'+year+' Data/data_next_1_station_'+str(year)+'_dm.pickle',"wb")
        cPickle.dump(dataset, pickle_out)
        pickle_out = open('D:/Wallace/Wallace-Github/data_preprocess/'+year+' Data/test_data_next_1_station_'+str(year)+'_dm.pickle',"wb")
        cPickle.dump(dataset, pickle_out)'''
    return dataset


    '''
    #apply dataframe to all steps
    pickle_in = open('D:/Wallace/Wallace-Github/data_preprocess/'+year+' Data/data_next_1_station_'+str(year)+'_dm.pickle',"rb")
    dataset_dm = cPickle.load(pickle_in)
    dataset_dm.to_csv(r'D:/Wallace/Wallace-Github/data_preprocess/'+year+' Data/data_next_1_station_'+str(year)+'_dm.txt')

    n_prediction_steps_range = list(range(2,11))

    for i in tqdm(range(0,len(n_prediction_steps_range))):
        n_prediction_steps = n_prediction_steps_range[i]
        
        pickle_in = open("../data/"+str(year)+"/DL_data_next_"+str(n_prediction_steps)+"_station_"+str(year)+".pickle","rb")
        dataset_2 = cPickle.load(pickle_in)
        dataset_2['delay_type_2'] = dataset_dm['delay_type_2']
        dataset_2['delay_type_2_desc'] = dataset_dm['delay_type_2_desc']
        dataset_2['delay_type_3'] = dataset_dm['delay_type_3']
        dataset_2['delay_type_3_desc'] = dataset_dm['delay_type_3_desc']
        dataset_2['delay_type_4'] = dataset_dm['delay_type_4']
        dataset_2['delay_type_4_desc'] = dataset_dm['delay_type_4_desc']
        dataset_2['delay_type_5'] = dataset_dm['delay_type_5']
        dataset_2['delay_type_5_desc'] = dataset_dm['delay_type_5_desc']
        
        pickle_out = open("../data/"+str(year)+"/DL_data_next_"+str(n_prediction_steps)+"_station_"+str(year)+"_dm.pickle",'wb')
        cPickle.dump(dataset_2, pickle_out)
    '''
    

