import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import json

from monolith import MelanomaFeatureCollector
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class MelanomaDatasetProcessor:
    def __init__(self, benign_path: str = "data/benign",
                 malignant_path: str = "data/malignant"):
        """
        Initialize the dataset processor

        Args:
            benign_path: Path to directory containing benign lesion images
            malignant_path: Path to directory containing malignant lesion images
        """
        self.benign_path = Path(benign_path)
        self.malignant_path = Path(malignant_path)
        self.feature_collector = MelanomaFeatureCollector()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess single image"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

    async def process_single_image(self, image_path: Path, label: int) -> Tuple[np.ndarray, int]:
        """Process single image and extract features"""
        try:
            # Load image
            image = self._load_image(image_path)

            # Extract features
            features = await self.feature_collector.collect_features(image)

            # Convert features to vector
            feature_vector = features.to_vector()

            return feature_vector, label

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None

    async def prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset from images

        Returns:
            Tuple containing features array and labels array
        """
        try:
            # Get all image paths
            benign_images = list(self.benign_path.glob("*.jpg")) + \
                            list(self.benign_path.glob("*.png"))
            malignant_images = list(self.malignant_path.glob("*.jpg")) + \
                               list(self.malignant_path.glob("*.png"))

            self.logger.info(f"Found {len(benign_images)} benign and "
                             f"{len(malignant_images)} malignant images")

            # Process all images
            features_list = []
            labels_list = []

            # Process benign images
            for img_path in tqdm(benign_images, desc="Processing benign images"):
                result = await self.process_single_image(img_path, 0)
                if result is not None:
                    feature_vector, label = result
                    features_list.append(feature_vector)
                    labels_list.append(label)

            # Process malignant images
            for img_path in tqdm(malignant_images, desc="Processing malignant images"):
                result = await self.process_single_image(img_path, 1)
                if result is not None:
                    feature_vector, label = result
                    features_list.append(feature_vector)
                    labels_list.append(label)

            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(labels_list)

            return X, y

        except Exception as e:
            self.logger.error(f"Error preparing dataset: {str(e)}")
            raise


class MelanomaTrainer:
    def __init__(self, random_state: int = 42):
        """
        Initialize the trainer

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='tanh',
            solver='adam',
            max_iter=1000,
            random_state=random_state,
            learning_rate='adaptive'
        )
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the model and evaluate performance

        Args:
            X: Feature array
            y: Label array

        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.logger.info("Training model...")
            self.model.fit(X_train_scaled, y_train)

            # Evaluate performance
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)

            # Calculate sensitivity and specificity
            y_pred = self.model.predict(X_test_scaled)
            TP = np.sum((y_test == 1) & (y_pred == 1))
            TN = np.sum((y_test == 0) & (y_pred == 0))
            FP = np.sum((y_test == 0) & (y_pred == 1))
            FN = np.sum((y_test == 1) & (y_pred == 0))

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'sensitivity': sensitivity,
                'specificity': specificity
            }

            self.logger.info(f"Training complete. Metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def save_model(self, path: str):
        """Save trained model and scaler"""
        import joblib

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)

        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str):
        """Load trained model and scaler"""
        import joblib

        saved_objects = joblib.load(path)

        trainer = cls()
        trainer.model = saved_objects['model']
        trainer.scaler = saved_objects['scaler']

        return trainer