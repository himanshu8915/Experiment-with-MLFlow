from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd 
import mlflow

import dagshub
dagshub.init(repo_owner='himanshu8915', repo_name='Experiment-with-MLFlow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/himanshu8915/Experiment-with-MLFlow.mlflow")

#load the dataset for breast cancer
data=load_breast_cancer()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name='target')

#split the data into training and testing sets 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#creating the randomforest classifier model
rf=RandomForestClassifier(random_state=42)

#definingthe parameter grid for grid search cv
param_grid={
    'n_estimators':[10,50,100],
    'max_depth':[None,10,20,30]
}

#apply grid search cv
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)

#run without mlflow from here
# grid_search.fit(x_train,y_train)

# #display best parameter and best score
# best_param=grid_search.best_params_
# best_score=grid_search.best_score_

# print(best_param)
# print(best_score)


mlflow.set_experiment('breast-cancer-rf-hp')

with mlflow.start_run() as parent:
    grid_search.fit(x_train,y_train)

    #log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as Child:
          mlflow.log_params(grid_search.cv_results_['params'][i])
          mlflow.log_metric("accuracy",grid_search.cv_results_["mean_test_score"][i])


    #display the best parameters and the best score
    best_params=grid_search.best_params_
    best_score=grid_search.best_score_


    #log params
    mlflow.log_params(best_params)

    #log metrics
    mlflow.log_metric("accuracy",best_score)

    #mlflow.log_metric - for single value
    #mlflow.log_metrics- expects dictionary

    #log the trainig data
    train_df=x_train.copy()
    train_df['target']=y_train

    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")

    #log the testing data
    test_data=x_test.copy()
    test_data['target']=y_test

    test_data=mlflow.data.from_pandas(test_data)
    mlflow.log_input(test_data,"testing")

    #log the source code
    mlflow.log_artifact(__file__)

    #log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_,"random-forest")

    #register the model
    mlflow.register_model("runs:/5f3526f05923408fac5a0ea9d6be2b97/random-forest","Model1")

    #set tags
    mlflow.set_tags({"author":"Himanshu"})
    #similarly here , mlflow.set_tag("author","Himanshu") for one tag but mlflow.set_tags(needs a dict input) for multiple tags

    print(best_params)
    print(best_score)


