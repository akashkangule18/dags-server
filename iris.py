import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import mlflow.sklearn



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
from sklearn.ensemble import RandomForestClassifier
X_train = train_data
X_test = test_data

mlflow.set_experiment('iris')
with mlflow.start_run():

    rf = RandomForestClassifier(n_estimators = 50,
                            min_samples_split = 2,
                                max_depth = 3
                            )

    rf.fit(X_train,y_train)

    # model evalution
    y_pred = rf.predict(X_test)

    from sklearn.metrics import accuracy_score,precision_score, recall_score

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('n_estimator',50)
    mlflow.log_param('min_sample_split',2)
    mlflow.log_param('max_depth',3)

    


    print('accuracy : ', accuracy)
