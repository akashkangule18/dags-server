import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
dagshub.init(repo_owner='akashkangule18', repo_name='dags-server', mlflow=True)



df = sns.load_dataset('iris')

# aplying label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# data spliting 
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, random_state = 42, test_size =0.2)

y_train = train_data['species'].values
y_test = test_data['species'].values

# # deleting output column before transformation
train_data.drop(columns=['species'],inplace = True)
test_data.drop(columns=['species'],inplace = True)

# model building
from sklearn.tree import DecisionTreeClassifier
X_train = train_data
X_test = test_data

mlflow.set_experiment('iris-decision-tree')
with mlflow.start_run():

    dt = DecisionTreeClassifier( max_depth = 1,
                                max_features=0.2,
                                min_impurity_decrease=0.1
                            )

    dt.fit(X_train,y_train)

    # model evalution
    y_pred = dt.predict(X_test)

    from sklearn.metrics import accuracy_score,precision_score, recall_score

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,average='weighted')
    recall = recall_score(y_test,y_pred,average='macro')


    # metrics
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_metric('precision',precision)
    mlflow.log_metric('recall',recall)

    # params
    mlflow.log_param('max_features',0.2)
    mlflow.log_param('max_depth',1)
    mlflow.log_param('min_impurity_decrease',0.1)

    # artifacts
    mlflow.log_artifact(__file__)

    # model
    mlflow.sklearn.log_model(dt, 'DecisionTreeClassifier')




    print('accuracy : ', accuracy)
    print('precision :', precision)
    print('recall: ',recall)
