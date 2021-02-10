import argparse
import os
import joblib

import pandas
from sklearn import linear_model, tree, ensemble
from sklearn import metrics

import config
import model_dispacher

def run(fold,model):

    training_file = pandas.read_csv(config.TRAINING_FILE)
    
    train = training_file[training_file.kfold != fold].reset_index(drop = True)
    test = training_file[training_file.kfold == fold].reset_index(drop = True)
    
    X_train = train.drop(['Activity'], axis = 1).values
    y_train = train.Activity.values
    
    X_test = test.drop(['Activity'], axis = 1).values
    y_test = test.Activity.values
    
    classifier = model_dispacher.models[model]

    classifier.fit(X_train,y_train)

    predictions = classifier.predict(X_test)

    print("accuracy:", metrics.accuracy_score(y_test,predictions))

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    #parser.add_argument("--fold",type = str)
    parser.add_argument("--model",type = str)

    args = parser.parse_args()

    run(
        fold = 5,
        model = args.model
    )
