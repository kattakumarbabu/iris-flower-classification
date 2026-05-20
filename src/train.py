# src/train.py
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_pipeline():
    print("--- Phase 1: Loading Dataset ---")
    # Path to the data file
    data_path = os.path.join('data', 'Iris.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find Iris.csv at {data_path}. Please place your file there.")
        
    df = pd.read_csv(data_path)
    
    print("--- Phase 2: Preprocessing Data ---")
    # Drop 'Id' as it is just a sequence number and doesn't provide predictive power
    X = df.drop(columns=['Id', 'Species'])
    y = df['Species']
    
    # Split the data: 80% for training the model, 20% for testing its accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("--- Phase 3: Training the Model ---")
    # Using a Random Forest Classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    print("--- Phase 4: Evaluating Model ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete. Test Set Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, y_pred))
    
    print("--- Phase 5: Saving Model Object ---")
    # Make sure the models directory exists
    os.makedirs('models', exist_ok=True)
    model_save_path = os.path.join('models', 'iris_model.pkl')
    
    # Save the model to a file using pickle so our web app can load it later
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model successfully saved to {model_save_path}")

if __name__ == '__main__':
    train_pipeline()