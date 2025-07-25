import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='himanshu8915', repo_name='Experiment-with-MLFlow', mlflow=True)


#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("https://dagshub.com/himanshu8915/Experiment-with-MLFlow.mlflow")

#load data
wine= load_wine()
x=wine.data
y=wine.target

# Train test split 
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.10,random_state=42)

# Define the params for RF model
max_depth=10
n_estimators=5


#Mention your experiment name so it does not go to default
mlflow.autolog()  #set this autolog to log all things automatically
mlflow.set_experiment('Exp-1') #this will create a experiemt if it is not there in ui

#other way is create from ui and then pass expId in the start function

with mlflow.start_run():  #experiment_id=346799096807584807
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(x_train,y_train)

    y_pred=rf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)


    #creating a confusion matrix plot
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    #save the plot
    plt.savefig('confusion-matrix.png')

   #log the artifacts using mlflow
    # we need to log explicitely for file 
    mlflow.log_artifact(__file__) # to log the file in which we are working, also we can do this by git but we can do this mlflow also if needed
     
    # set tags
    mlflow.set_tags({"Author":"himanshu","Project":"Wine classification"})


    print(accuracy) 