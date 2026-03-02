import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[['study_hours', 'attendance', 'internal_marks']]
y = data['result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model + accuracy
pickle.dump((model, accuracy), open("model.pkl", "wb"))

print(f"Model trained successfully! Accuracy: {accuracy*100:.2f}%")