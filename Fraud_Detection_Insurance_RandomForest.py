"""
@author: Dr Yen Fred WOGUEM 

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


import pandas as pd
import numpy as np

# seed
np.random.seed(42)


data_size = 20000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, data_size),
    'income': np.random.randint(20000, 100000, data_size),
    'policy_amount': np.random.randint(5000, 50000, data_size),
    'num_claims': np.random.randint(0, 5, data_size),
    'fraudulent': np.random.choice([0, 1], size=data_size, p=[0.7, 0.3])  # 40% de fraude
})

# Save data
#data.to_csv('assurance_data.csv', index=False)




# Data processing

X = data.drop('fraudulent', axis=1)  # Features
y = data['fraudulent']  # Target

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calssification model
model = RandomForestClassifier(max_depth=6,
                               class_weight="balanced")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))










