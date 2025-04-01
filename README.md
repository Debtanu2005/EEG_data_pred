
# EEG Prediction

## Overview
This project focuses on EEG (Electroencephalogram) signal analysis and classification using Machine Learning and Deep Learning techniques. The dataset used is the Epileptic Seizure Recognition dataset, which helps in distinguishing seizure-related brain activity from normal activity.

## Dataset
The dataset consists of EEG signals stored in CSV format, with multiple time-series features representing brain activity. The target variable (`y`) represents different classes of brain activity.

## Features
- **Signal Preprocessing**: Fourier Transform, Outlier Removal, Standardization, Normalization
- **Machine Learning Models**:
  - Decision Tree Classifier
  - Random Forest Classifier
- **Deep Learning Model**:
  - Bidirectional LSTM-based Neural Network

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow
```

## Usage
### 1. Data Loading
```python
import pandas as pd

df = pd.read_csv("Epileptic Seizure Recognition.csv")
print(df.head())
```

### 2. Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_standard = sc.fit_transform(df.drop(["y"], axis=1))
```

### 3. Model Training
#### Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=8, min_samples_split=15)
model.fit(x_train, y_train)
```
#### Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(x_train, y_train)
```

### 4. Deep Learning Model (LSTM)
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input

input_layer = Input(shape=(178,))
dense_layer = Dense(100, activation='relu')(input_layer)
lstm_layer = Bidirectional(LSTM(100, return_sequences=False))(dense_layer)
output_layer = Dense(5, activation='softmax')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 5. Model Evaluation
```python
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)
```

## Results
The trained models achieved significant accuracy in classifying EEG signals. The Random Forest model performed well, while the LSTM-based deep learning model achieved an accuracy of up to 68%.

## Contributing
Feel free to fork this repository and contribute improvements!


