import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5000")

#load data
wine= load_wine()
x=wine.data
y=wine.target

# Train test split 
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.10,random_state=42)

# Define the params for RF model
max_depth=10
n_estimators=10


#Mention your experiment name so it does not go to default
mlflow.set_experiment('Exp-1')

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(x_train,y_train)

    y_pred=rf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_metric('max_depth',max_depth)
    mlflow.log_metric('n_estimators',n_estimators)

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
    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact(__file__) # to log the file in which we are working, also we can do this by git but we can do this mlflow also if needed


    print(accuracy)