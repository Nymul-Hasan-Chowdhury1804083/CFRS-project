from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())

print(df.size)

print(df.shape)

print(df.columns)

df['crops'].unique()

df['Fertilizer'].unique()

df['crops'].value_counts()

df['Fertilizer'].value_counts()

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph','moisture', 'soil temp']].values
y = df[['crops', 'Fertilizer']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

print(X_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

print(X_train)

base_classifier = RandomForestClassifier()

base_classifier.fit(X_train,y_train)

multi_target_model = MultiOutputClassifier(base_classifier, n_jobs=-1)

multi_target_model.fit(X_train, y_train)

y_pred = multi_target_model.predict(X_test)

# Calculate accuracy for each output
accuracy_crop = accuracy_score(y_test[:, 0], y_pred[:, 0])
accuracy_fertilizer = accuracy_score(y_test[:, 1], y_pred[:, 1])

print("Accuracy for crop prediction:", accuracy_crop*100)
print("Accuracy for fertilizer prediction:", accuracy_fertilizer*100)


import pickle
pickle.dump(base_classifier,open('model.pkl','wb'))
pickle.dump(ms,open('minmaxscaler.pkl','wb'))
pickle.dump(sc,open('standscaler.pkl','wb'))