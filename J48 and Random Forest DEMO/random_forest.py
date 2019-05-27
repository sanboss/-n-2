import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


features = pd.read_csv('C:/Users/user/Downloads/Breast_Cancer.csv')
print features.describe()

features = pd.get_dummies(features)
# print features.iloc[:,:].head(5)

x= features.iloc[:,:-1]
labels = np.array(features['class'])
features= features.drop('class', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(x, labels, test_size = 0.95, random_state = 42)
print('Training Features Shape:', len(train_features))
print('Testing Features Shape:', len(test_features))

rf = RandomForestRegressor(n_estimators = 1, max_depth = 5,  random_state = 42)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
print "confusion_matrix"
print(confusion_matrix(test_labels,predictions.round()))
print "classification_report"
print(classification_report(test_labels,predictions.round()))
print "accuracy_score"
print(accuracy_score(test_labels, predictions.round()))


errors = abs(np.mean(test_labels) - np.mean(predictions))
print('Mean Absolute Error:', round(np.mean(errors),8), 'degrees.')
mape = 100*(np.mean(errors)/np.mean(test_labels))
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Import tools needed for visualization





os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
tree = rf.estimators_[1]
export_graphviz(tree, out_file = 'tree.dot',\
                feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('C:/Users/user/Downloads/tree.png')


importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2))
            for feature, importance in zip(feature_list, importances)]
feature_importances = \
    sorted(feature_importances, key = lambda x: x[1], reverse = True)
for pair in feature_importances :
    print('Variable: {:20} Importance: {}'.format(*pair));

plt.style.use('fivethirtyeight')
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable');
plt.title('Variable Importances');
# plt.show()

