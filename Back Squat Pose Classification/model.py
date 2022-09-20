import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import json
import pandas as pd
import seaborn as sns
from sklearn import metrics



# Import names of json files into list
numFiles = 46
files=[]
for i in range(numFiles):
    files.append('Side Training Data/squat_video'+str(i+1)+'.json')

# Add all coordinates from JSON file to dictionary
data = {}
def data_collector(files, data):
    for file in files:
        with open(file, 'r') as infile:
            temp = json.load(infile)
            for key, list_of_coordinates in temp.items():
                if(key != 'value'):
                    for coordinates in list_of_coordinates:
                        i = 0
                        for num in coordinates:
                            i += 1
                            try:
                                data[key+str(i)].append(num)
                            except KeyError:
                                data[key+str(i)] = [num]
                        if(key == 'z'):
                            try:
                                data['type'].append(temp['value'])
                            except KeyError:
                                data['type'] = [temp['value']]
    return pd.DataFrame.from_dict(data)

df = data_collector(files, data)

print(df)

# Separate dataframe into independent and dependent variables
y = df['type']
x = df.drop('type', axis = 'columns')

# Split up data into testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Train algorithm
range_k = range(1,70)
scores = {}
scores_list = []
for k in range_k:
   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(x_train, y_train)
   y_pred = classifier.predict(x_test)
   scores[k] = metrics.accuracy_score(y_test,y_pred)
   scores_list.append(metrics.accuracy_score(y_test,y_pred))
print()
print(k)
result = metrics.confusion_matrix(y_test, y_pred)
print()
print("Confusion Matrix:")
print(result)
result1 = metrics.classification_report(y_test, y_pred)
print()
print("Classification Report:",)
print (result1)
plt.plot(range_k,scores_list)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.show()


# TESTING
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

files = ['Side Training Data/squat_video47.json']

data = {}
test_df = data_collector(files,data)

y_test = test_df['type']
x_test = test_df.drop('type', axis = 'columns')

predictions = classifier.predict(x_test)

y_test_list = test_df['type'].values.tolist()

if(y_test_list[0] == 1):
    percent_accuracy = sum(predictions)/len(predictions)
else:
    percent_accuracy = (len(predictions) - sum(predictions))/len(predictions)

print('Percent of frames accurately classified: ',percent_accuracy)

print()

print('Predictions: ',predictions)
print()
print('Actual: ',y_test_list)








