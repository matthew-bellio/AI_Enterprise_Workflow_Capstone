#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(tag,mae,mse,r2,adj_r2,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join("logs", "train-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file
    header = ['unique_id','timestamp','tag', 'period', 'mae', 'mse', 'r2', 'adj_r2','model_version',
              'model_version_note','runtime']

    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), tag, mae, mse, r2, adj_r2,
                            MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)

def update_predict_log(country, y_pred, y_proba, target_date, MODEL_VERSION, runtime, test=False):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file
    header = ['unique_id','timestamp','country', 'y_pred','y_proba','target_date','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(), time.time(), country, y_pred, y_proba,target_date, MODEL_VERSION, runtime])
        writer.writerow(to_write)

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    from model import MODEL_VERSION, MODEL_VERSION_NOTE

    ## train logger
    update_train_log(tag = "unit_test", period = ("2017-11-01", "2019-06-30"), mae = 0, mse = 0, r2 = 0, adj_r2 = 0, model_version = 0.0, model_version_note = "test model",runtime = "00:00:01")

    ## predict logger
    update_predict_log(country='test', y_pred = [0], y_proba = None, target_date = '2019-01-05', runtime = "00:00:01", model_version = 0.0)
