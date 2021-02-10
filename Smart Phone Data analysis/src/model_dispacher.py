from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble

models = {
    
    "random_forest_classifier": ensemble.RandomForestClassifier(),
    "random_forest_regressor": ensemble.RandomForestRegressor(),
    "decision_tree_classifier": tree.DecisionTreeClassifier(),
    "decision_tree_regressor": tree.DecisionTreeRegressor(),
    "logistic_regression": linear_model.LogisticRegression(),
    "linear_regression": linear_model.LinearRegression()
    
}

