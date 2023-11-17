import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# File path to your CSV file
file_path = ''

# Define chunk size
chunk_size = 5000

# Initialize an empty DataFrame to hold chunks
data = pd.DataFrame()

# Read the file in chunks
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Remove leading/trailing whitespaces from column names
    chunk.columns = chunk.columns.str.strip()

    # Replace inf/-inf with NaN
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    chunk.dropna(inplace=True)

    data = pd.concat([data, chunk])

# Basic preprocessing
X = data.drop('Label', axis=1)
y = data['Label']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting and evaluating the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
