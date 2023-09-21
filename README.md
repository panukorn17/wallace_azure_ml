# Wallace Delay Prediction

## Introduction
Reactionary delays that propagate from a primary source throughout train journeys are an immediate concern for British railway systems. Complex non-linear interactions between various spatiotemporal variables govern the propagation of these delays which can avalanche throughout railway network causing further severe disruptions. This code is an implementation from the following [paper](https://www.tandfonline.com/doi/abs/10.1080/15472450.2020.1858822). This introduces several machine learning (ML) techniques alongside data preprocesses to create a framework that predicts key performance indicators (KPIs), reactionary arrival delay, reactionary departure delay, dwell time and travel time. The frameworks in this paper provide greater accuracy in predicting KPIs through state-of-the-art ML models compared to existing delay prediction systems. 

## Install Requirements
Run:  
```pip install -r requirements.txt```  
This will take care of installing all required dependencies  

## HSP Data
The data used in for this code is from [Darwin's HSP Platform](https://wiki.openraildata.com/index.php/HSP). Specifically, the data to be preprocessed and trained on can be requested by a HTTP POST call of the 'serviceDetails' API.  

```
POST https://hsp-prod.rockshore.net/api/v1/serviceMetrics HTTP/1.1
Authorization: Basic {Base64("email:password")}
Content-Type: application/json
Host: hsp-prod.rockshore.net
Content-Length: 178

{
    "from_loc":"DID",
    "to_loc":"PAD",
    "from_time":"0700",
    "to_time":"0800",
    "from_date":"2016-07-01",
    "to_date":"2016-08-01",
    "days":"WEEKDAY"
}
```

## Preprocessing the data:
To preprocess the data, run `'data_preprocess.py'`. This will create a directory where preprocessed data is dumped.

## Training the model
To reproduce the training of our model, run the `'training.ipynb'`. This will train a DNN model that predicts the arrival delay of a train from one stop ahead.