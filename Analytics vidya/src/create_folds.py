from sklearn import model_selection
import pandas
import argparse

import config 

class folding:
    def __init__(self,data):
        self.data = data
    
    def Stratified_fold(self):
        
        self.data = data.sample(frac = 1).reset_index(drop = True)

        y = self.data.Response.values

        kf = model_selection.StratifiedKFold(n_splits = 20)

        for f, (t_, v_) in enumerate(kf.split(X=data, y=y)):
            self.data.loc[v_, 'kfold'] = f
        
        self.data.to_csv(config.FOLDS,index = False)

    def Kfold(self):
        self.data = data.sample(frac = 1).reset_index(drop = True)

        y = self.data.Activity.values

        kf = model_selection.StratifiedKFold(n_splits = 20)

        for f, (t_, v_) in enumerate(kf.split(X=data, y=y)):
            self.data.loc[v_, 'kfold'] = f
        
        self.data.to_csv(config.FOLDS,index = False)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fold",type = str)

    args = parser.parse_args()

    data = pandas.read_csv(config.TRAINING_FILE)
    k = folding(data)

    if args.fold == "Stratified_fold":
        k.Stratified_fold()
    elif args.fold == "Kfold":
        k.Kfold()
        
    