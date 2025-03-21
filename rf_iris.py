import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Load the Iris dataset from CSV
    iris = pd.read_csv('iris.csv')
    X = iris.drop('Species', axis=1)
    y = iris['Species']

    # Split the dataset (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def save_model(model, filename='rf_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    model = train_model()
    save_model(model)
