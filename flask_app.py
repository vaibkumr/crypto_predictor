import json
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
import time
import datetime
import requests
import pandas as pd
import pickle

# http://127.0.0.1:5000/?model=dt&date=2018_1_1
def generate_features(timestamp,price_data):
    features=pd.DataFrame()
    features['p_2017'],features['p_2016'],features['p_2015'],features['p_2014']=[0,0,0,0]
    for year in [1,2,3,4]:
        key='p_201'+str(8-year)
        year_stamp=timestamp-(365*86400*year)
        features.at[0,key]=price_data.loc[year_stamp].price
    return features


def predict_price(model,timestamp):
    file_name="data_.obj"
    cur_dir=os.path.dirname(__file__)
    file=os.path.join(cur_dir,file_name)
    with open(file,'rb') as handle:
        price_data=pickle.load(handle)
    features=generate_features(timestamp,price_data)
    prediction=model.predict(features)
    return "{:.2f}".format(float(prediction))

def get_epoch(date):
    _date=date.split("_")
    year,month,day=int(_date[0]),int(_date[1]),int(_date[2])
    timestamp=datetime.datetime(year,month,day).timestamp()
    return int(timestamp)

def dt(timestamp):
    price_dict={}
    file_names=["decision_tree_3.obj","decision_tree_5.obj","decision_tree_7.obj"]
    depths=[3,5,7]
    cur_dir=os.path.dirname(__file__)
    for file_name,depth in zip(file_names,depths):
        file=os.path.join(cur_dir,file_name)
        with open(file,'rb') as handle:
            obj=pickle.load(handle)
            key="price(max_depth:"+str(depth)+")"
            price_dict[key]=predict_price(obj,timestamp)
    return price_dict

def rr(timestamp):
    price_dict={}
    file_names=["ridge_regression_0.obj","ridge_regression_10.obj","ridge_regression_20.obj"]
    alphas=[0,10,20]
    cur_dir=os.path.dirname(__file__)
    for file_name,alpha in zip(file_names,alphas):
        file=os.path.join(cur_dir,file_name)
        with open(file,'rb') as handle:
            obj=pickle.load(handle)
            key="price(alpha:"+str(alpha)+")"
            price_dict[key]=predict_price(obj,timestamp)
    return price_dict

def mlp_nn(timestamp):
    price_dict={}
    file_names=["neural_mlp_5__0.0001.obj","neural_mlp_10__0.0001.obj","neural_mlp_20__0.0001.obj"]
    layers=[5,10,20]
    cur_dir=os.path.dirname(__file__)
    for file_name,layer in zip(file_names,layers):
        file=os.path.join(cur_dir,file_name)
        with open(file,'rb') as handle:
            obj=pickle.load(handle)
            key="price(hidden layers:"+str(layer)+")"
            price_dict[key]=predict_price(obj,timestamp)

    return price_dict

def rfr(timestamp):
    price_dict={}
    file_names=["random_forest_f.obj"]
    cur_dir=os.path.dirname(__file__)
    for file_name in file_names:
        file=os.path.join(cur_dir,file_name)
        with open(file,'rb') as handle:
            obj=pickle.load(handle)
            key="price:"
            price_dict[key]=predict_price(obj,timestamp)
    return price_dict

app = Flask(__name__)
CORS(app)
@app.route("/",methods=['GET'])
def return_price():
    date=request.args.get('date')
    model=request.args.get('model')
    if(model=='dt'):
        timestamp=get_epoch(date)
        return jsonify(dt(timestamp))
    elif(model=='rr'):
        timestamp=get_epoch(date)
        return jsonify(rr(timestamp))
    elif(model=='mlp_nn'):
        timestamp=get_epoch(date)
        return jsonify(mlp_nn(timestamp))
    elif(model=='rfr'):
        timestamp=get_epoch(date)
        return jsonify(rfr(timestamp))
    else:
        return "test"

if __name__ == "__main__":
    app.run()
