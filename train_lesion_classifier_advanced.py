import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class LesionClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_model = None
        self.selected_features = None
        
        # Define malignant lesions
        self.malignant_types = [
            'melanoma',
            'basal cell carcinoma',
            'squamous cell carcinoma'
        ]

    def load_and_preprocess_data(self, csv_path):
        # Load data
        print("Loading data from:", csv_path)
        df = pd.read_csv(csv_path)
        X = df.drop(['label', 'image_path'], axis=1)
        
        # Convert to binary classification (benign/malign)
        y = df['label'].apply(lambda x: 1 if x in self.malignant_types else 0)
        
        print("\nClass distribution:")
        print("Benign lesions:", sum(y == 0))
        print("Malignant lesions:", sum(y == 1))
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        return X, y

    def perform_feature_selection(self, X, y):
        print("\nPerforming feature selection...")
        # Use RandomForest for feature selection
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = selector.fit(X, y)
        
        # Select features based on importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        threshold = np.mean(selector.feature_importances_)
        self.feature_selector = SelectFromModel(selector, prefit=True, threshold=threshold)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [f for f, selected in zip(self.feature_names, selected_mask) if selected]
        
        print("\nSelected features:", self.selected_features)
        self.plot_feature_importance(feature_importance)
        
        return self.feature_selector.transform(X)

    def plot_feature_importance(self, feature_importance):
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()

    def train_model(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Apply SMOTE for class balancing
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print("\nAfter SMOTE - samples per class:")
        print("Benign:", sum(y_train_balanced == 0))
        print("Malignant:", sum(y_train_balanced == 1))
        
        # Define model and parameters
        model = RandomForestClassifier(random_state=42)
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print(f"\nTraining RandomForest with GridSearchCV...")
        
        # Perform GridSearch
        grid_search = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate model
        self.best_model = grid_search.best_estimator_
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        print(f"\nBest Parameters:", grid_search.best_params_)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                 target_names=['Benign', 'Malignant']))
        
        # Plot ROC curve
        self.plot_roc_curve(y_test, y_pred_proba)
        
        # Save confusion matrix plot
        self.plot_confusion_matrix(y_test, y_pred)
        
        return X_test, y_test

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    def save_model(self):
        # Save model and preprocessing objects
        print("\nSaving model and preprocessing objects...")
        joblib.dump(self.best_model, 'best_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        joblib.dump(self.feature_selector, 'feature_selector.joblib')
        
        # Save selected features list
        with open('selected_features.txt', 'w') as f:
            f.write('\n'.join(self.selected_features))
        print("All files saved successfully!")

def main():
    classifier = LesionClassifier()
    
    # Load and preprocess data
    print("Starting binary lesion classification (benign vs malignant)...")
    X, y = classifier.load_and_preprocess_data('asymmetry_module/lesion_features.csv')
    
    # Perform feature selection
    X_selected = classifier.perform_feature_selection(X, y)
    
    # Train and evaluate model
    X_test, y_test = classifier.train_model(X_selected, y)
    
    # Save the model
    classifier.save_model()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
