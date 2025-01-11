import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Drop the image_path column and any other non-feature columns
    X = df.drop(['label', 'image_path'], axis=1)
    
    # Convert labels to numerical values
    le = LabelEncoder()
    le.fit(df['label'])
    joblib.dump(le, 'label_encoder.joblib')
    y = le.transform(df['label'])
    
    return X, y

# Train and evaluate models
def train_models(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        print(f"\n{name} Results:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': model.score(X_test_scaled, y_test)
        }
    
    # Save the best model and scaler
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Save the model and scaler
    joblib.dump(best_model, 'best_lesion_classifier.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    return best_model, scaler

def main():
    csv_path = 'asymmetry_module/lesion_features.csv'
    print("Loading data...")
    X, y = load_data(csv_path)
    
    print("\nFeature names:", list(X.columns))
    print("\nNumber of features:", X.shape[1])
    print("Number of samples:", X.shape[0])
    print("\nUnique classes:", len(np.unique(y)))
    
    best_model, scaler = train_models(X, y)
    print("\nModel and scaler saved successfully!")

if __name__ == "__main__":
    main()
