#Shit tier ML application for bitcoin price prediction

## Introduction
Crappy model for bitcoin price prediction using sklearn module's regressor and decision trees. The model is trained over past 4 years data fetched using the coindesk API. The trained model objects are stored in /trained_objects as pickle formatted objects. 
flask is used to deploy the models online and retrieve the prediction.

## ML Model
The depended variables are :-
- Price on xx/xx/2014
- Price on xx/xx/2015
- Price on xx/xx/2016
- Price on xx/xx/2017

The independent variable (to be predited) is:-
- Price on xx/xx/2018

Hence, we are limiting the predictions for year 2018, anything after 12/30/2018 won't be, rather can't be predicted using this crappy model. 

## Flask app
flask_app.py is used to deploy the model online and retrieve prices by calculating predictions over the already trained models (no dynamic data)
flask webapp arguments :-
- model : dt, rr, mll_rr, rfr
- data: date in MM/DD/YYYY format for prediction

## Drawbacks and future developments
- Temporal data yet i've used randomized train-test sample which gives insights into the future
- Not enough dependend variables, dates give much more insight than just the UNIX epoch, columns shall be divided into day of month, day of year, day of week etc
- For other cryptos than bitcoin, bitcoin price should be one of the dependted variable, based on the generaly community phrase "bitcoin is the highway, other cryptos are the roads'
- No feature engineering whatsoever was done, tch tch tch...
- Many more...

