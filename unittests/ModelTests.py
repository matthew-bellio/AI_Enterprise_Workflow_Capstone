#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from model import *

#MODEL_PATH = path.join(wdir,'models')
#DATA_PATH = path.join(wdir,'cs-train')
#LOG_PATH = path.join(wdir,'logs')

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(os.path.join("data","cs-train"),test=True)
        trained_model = os.path.join("models", "sl-eire-0_1.joblib")
        self.assertTrue(os.path.exists(trained_model))
        #self.assertTrue(os.path.exists(os.path.join("models", "sl-eire-0_1.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """

        ## load the models and corresponding data
        all_data, all_models = model_load(country='eire')
        model = all_models['eire']

        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))


    def test_03_predict(self):
        """
        test the predict function input
        """

        ## ensure that a list can be passed
        country='eire'
        year='2018'
        month='05'
        day='01'

        result = model_predict(country, year, month, day, test=True)
        self.assertTrue(result['y_pred'] > 0.0)


### Run the tests
if __name__ == '__main__':
    unittest.main()
