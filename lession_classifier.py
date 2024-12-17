import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Union
import logging


class MelanomaClassifier:
    """
    Melanoma classification system based on shape, color and texture features
    as described in the paper.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Initialize the neural network classifier with architecture from paper
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers
            activation='tanh',  # Hyperbolic tangent activation
            solver='adam',  # Adam optimizer
            max_iter=1000,
            random_state=42,
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.2,  # Use 20% of training data for validation
        )

        self.scaler = StandardScaler()
        self._feature_groups = {
            'shape': [
                'asymmetry_index_1', 'asymmetry_index_2',
                'irregularity_A', 'irregularity_B',
                'irregularity_C', 'irregularity_D',
                'perimeter', 'area'
            ],
            'color': [
                'l_mean', 'a_mean', 'b_mean',
                'l_std', 'a_std', 'b_std',
                'mean_color_difference', 'max_color_difference'
            ],
            'texture': [
                'contrast', 'dissimilarity', 'homogeneity',
                'energy', 'correlation', 'ASM',
                'sum_variance', 'diff_variance',
                'sum_entropy', 'diff_entropy',
                'mean_intensity', 'std_intensity',
                'mean_gradient', 'std_gradient'
            ]
        }

    def _extract_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Extract features in the correct order for the model.

        Args:
            features: Dictionary containing feature values

        Returns:
            numpy.ndarray: Feature vector in correct order
        """
        feature_vector = []

        # Extract features in defined group order
        for group in ['shape', 'color', 'texture']:
            for feature_name in self._feature_groups[group]:
                feature_vector.append(features[feature_name])

        return np.array(feature_vector).reshape(1, -1)

    def train(self, X_train: Union[np.ndarray, List[Dict]], y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the classifier.

        Args:
            X_train: Training features (either array or list of feature dictionaries)
            y_train: Training labels (0 for benign, 1 for malignant)

        Returns:
            Dictionary containing training metrics
        """
        try:
            # Convert feature dictionaries to array if needed
            if isinstance(X_train, list):
                X_train = np.array([self._extract_feature_vector(x)[0] for x in X_train])

            # Scale features
            X_scaled = self.scaler.fit_transform(X_train)

            # Train model
            self.model.fit(X_scaled, y_train)

            # Calculate training metrics
            train_score = self.model.score(X_scaled, y_train)
            y_pred = self.model.predict(X_scaled)

            # Calculate metrics
            tp = np.sum((y_train == 1) & (y_pred == 1))
            tn = np.sum((y_train == 0) & (y_pred == 0))
            fp = np.sum((y_train == 0) & (y_pred == 1))
            fn = np.sum((y_train == 1) & (y_pred == 0))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            return {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'training_score': train_score
            }

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, features: Union[Dict[str, float], np.ndarray]) -> Dict[str, float]:
        """
        Classify a lesion using extracted features.

        Args:
            features: Dictionary of features or feature vector

        Returns:
            Dictionary containing classification results and probabilities
        """
        try:
            # Convert features to vector if dictionary
            if isinstance(features, dict):
                X = self._extract_feature_vector(features)
            else:
                X = features.reshape(1, -1)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get prediction and probabilities
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]

            return {
                'classification': 'malignant' if prediction == 1 else 'benign',
                'malignant_probability': float(probabilities[1]),
                'benign_probability': float(probabilities[0]),
                'prediction_confidence': float(max(probabilities))
            }

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on connection weights.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            # Get weights from first layer
            weights = np.abs(self.model.coefs_[0])

            # Calculate importance as mean absolute weight for each feature
            importance = np.mean(np.abs(weights), axis=1)

            # Normalize importance scores
            importance = importance / np.sum(importance)

            # Create feature importance dictionary
            feature_names = []
            for group in ['shape', 'color', 'texture']:
                feature_names.extend(self._feature_groups[group])

            return dict(zip(feature_names, importance))

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            raise