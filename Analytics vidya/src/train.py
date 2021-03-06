import pandas
import numpy 
from sklearn import metrics

import config
import model_dispacher

import argparse

def run(fold, model):
    training = pandas.read_csv("F:\\Analytics vidya\\input\\train_processed.csv")
    #testing = pandas.read_csv("F:\\Analytics vidya\\input\\test_processed.csv")

    train = training[training.kfold != fold].reset_index(drop = True)
    test = training[training.kfold == fold].reset_index(drop = True)
    
    X_train = train.drop(['source','ID','Response'],axis = 1).values
    y_train = train.Response.values
    
    X_test = test.drop(['source','ID','Response'], axis = 1).values
    y_test = test.Response.values
    classifier = model_dispacher.models[model]

    classifier.fit(X_train,y_train)

    predictions = classifier.predict(X_test)

    print("accuracy:", metrics.accuracy_score(y_test,predictions))

    testing_file = pandas.read_csv("F:\\Analytics vidya\\input\\test_processed.csv")

    testing_file.dropna(1,inplace=True)

    testing_file.drop(['source'],axis = 1,inplace = True)

    final_predictions = classifier.predict_proba(testing_file.values)

    return (final_predictions) 
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fold",type = float)
    parser.add_argument("--model",type = str)

    args = parser.parse_args()

    k = run(
        fold = args.fold,
        model = args.model
    )


    submissions = pandas.read_csv("F:\\Analytics vidya\\input\\sample_submission.csv") 
    p = []
    count = 0
    for i in k:
        submissions['Response'][count] = i[1]
        count = count + 1

    print(submissions.Response.unique())