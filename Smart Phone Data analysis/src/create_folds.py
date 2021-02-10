import pandas
from sklearn import model_selection

import config

if __name__ == "__main__":

    data = pandas.read_csv(config.TRAINING_FILE)
    data.drop(["source","subject"],axis = 1,inplace = True)

    data['kfold'] = -1

    data = data.sample(frac = 1).reset_index(drop = True)

    y = data.Activity.values

    kf = model_selection.StratifiedKFold(n_splits = 20)

    for f, (t_, v_) in enumerate(kf.split(X=data, y=y)):
        data.loc[v_, 'kfold'] = f
    
    data.to_csv("D:\\smart_phone_dataset\\archive\\project\\input\\folds.csv",index = False)