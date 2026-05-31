"""
Advanced ML Training Module for Fake News Detection
Implements multiple models and ensemble methods for improved accuracy
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import os
import sys
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers not available - skipping BERT models")


# Module-level class for pickle serialization
class EnsemblePredictor:
    """Ensemble predictor using voting from multiple models"""
    
    def __init__(self, models_data: Dict):
        self.models = models_data
    
    def predict(self, texts: list):
        """Ensemble prediction using voting"""
        predictions = []
        confidences = []
        
        for text in texts:
            pred_list = []
            conf_list = []
            
            # Get predictions from each model
            for name, model_info in self.models.items():
                model = model_info['model']
                vectorizer = model_info['vectorizer']
                
                try:
                    X_vec = vectorizer.transform([text])
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict(X_vec)[0]
                        proba = model.predict_proba(X_vec)[0]
                        conf = np.max(proba)
                    else:
                        pred = model.predict(X_vec.toarray())[0]
                        proba = model.predict_proba(X_vec.toarray())[0]
                        conf = np.max(proba)
                    
                    pred_list.append(pred)
                    conf_list.append(conf)
                except Exception as e:
                    logger.warning(f"Error in {name} prediction: {e}")
            
            # Voting ensemble
            if pred_list:
                unique, counts = np.unique(pred_list, return_counts=True)
                ensemble_pred = unique[np.argmax(counts)]
                ensemble_conf = np.mean(conf_list)
            else:
                ensemble_pred = 'real'
                ensemble_conf = 0.5
            
            predictions.append(ensemble_pred)
            confidences.append(ensemble_conf)
        
        return predictions, confidences


class AdvancedNewsDetector:
    """Advanced fake news detection with multiple models and ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.performance_metrics = {}
    
    def train_logistic_regression(self, X, y, X_test, y_test) -> Dict:
        """Train Logistic Regression model (fast and reliable baseline)"""
        logger.info("Training Logistic Regression Model...")
        
        # Multiple TF-IDF configurations for better feature extraction
        config = {
            'max_features': 5000,
            'ngram_range': (1, 3),  # Unigrams, bigrams, trigrams
            'max_df': 0.85,
            'min_df': 2,
            'strip_accents': 'unicode',
            'lowercase': True,
            'analyzer': 'word',
            'token_pattern': r'(?u)\b\w\w+\b',
        }
        
        vectorizer = TfidfVectorizer(**config, stop_words='english')
        X_vec = vectorizer.fit_transform(X)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LogisticRegression(
            max_iter=1500,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced',  # Handle class imbalance
            C=0.8,  # Regularization
        )
        model.fit(X_vec, y)
        
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        logger.info(f"✅ Logistic Regression Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': classification_report(y_test, y_pred)
        }
        
        return model, vectorizer, metrics
    
    def train_random_forest(self, X, y, X_test, y_test) -> Dict:
        """Train Random Forest model (better at capturing complex patterns)"""
        logger.info("Training Random Forest Model...")
        
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2,
            stop_words='english'
        )
        X_vec = vectorizer.fit_transform(X).toarray()  # Convert to dense for RF
        X_test_vec = vectorizer.transform(X_test).toarray()
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_vec, y)
        
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        logger.info(f"✅ Random Forest Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': classification_report(y_test, y_pred)
        }
        
        return model, vectorizer, metrics
    
    def train_gradient_boosting(self, X, y, X_test, y_test) -> Dict:
        """Train Gradient Boosting model (advanced ensemble method)"""
        logger.info("Training Gradient Boosting Model...")
        
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2,
            stop_words='english'
        )
        X_vec = vectorizer.fit_transform(X).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8,
            verbose=0
        )
        model.fit(X_vec, y)
        
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        logger.info(f"✅ Gradient Boosting Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': classification_report(y_test, y_pred)
        }
        
        return model, vectorizer, metrics
    
    def create_ensemble_predictor(self, models_data: Dict):
        """
        Create ensemble predictor that combines multiple models
        
        Args:
            models_data: Dictionary with trained models and vectorizers
        """
        return EnsemblePredictor(models_data)


def train_advanced_models(data_file: str = "fakereal_training_data.csv", 
                         output_dir: str = ".") -> Dict:
    """
    Train multiple advanced models and save them
    
    Args:
        data_file: Path to training data CSV
        output_dir: Directory to save models
    
    Returns:
        Dictionary with training results
    """
    
    # Check if data exists
    if not os.path.exists(data_file):
        logger.error(f"❌ {data_file} not found!")
        return None
    
    try:
        # Load and prepare data
        logger.info(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file).dropna()
        
        if len(df) == 0:
            logger.error("❌ No data in file")
            return None
        
        if 'text' not in df.columns or 'label' not in df.columns:
            logger.error("❌ Required columns 'text' and 'label' not found")
            return None
        
        X = df['text']
        y = df['label']
        
        # Data split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Train multiple models
        detector = AdvancedNewsDetector()
        
        # 1. Logistic Regression
        lr_model, lr_vec, lr_metrics = detector.train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        
        # 2. Random Forest
        rf_model, rf_vec, rf_metrics = detector.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # 3. Gradient Boosting
        gb_model, gb_vec, gb_metrics = detector.train_gradient_boosting(
            X_train, y_train, X_test, y_test
        )
        
        # Prepare models for ensemble
        models_data = {
            'logistic_regression': {'model': lr_model, 'vectorizer': lr_vec},
            'random_forest': {'model': rf_model, 'vectorizer': rf_vec},
            'gradient_boosting': {'model': gb_model, 'vectorizer': gb_vec},
        }
        
        # Create and test ensemble
        logger.info("\n🔄 Creating Ensemble Model...")
        ensemble = detector.create_ensemble_predictor(models_data)
        
        # Test ensemble on test set
        y_pred_ens, conf_ens = ensemble.predict(X_test.tolist())
        ensemble_accuracy = accuracy_score(y_test, y_pred_ens)
        logger.info(f"✅ Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        # Save all models
        logger.info("\n💾 Saving models...")
        joblib.dump(lr_model, os.path.join(output_dir, "model_fake_lr.pkl"))
        joblib.dump(lr_vec, os.path.join(output_dir, "vectorizer_fake_lr.pkl"))
        
        joblib.dump(rf_model, os.path.join(output_dir, "model_fake_rf.pkl"))
        joblib.dump(rf_vec, os.path.join(output_dir, "vectorizer_fake_rf.pkl"))
        
        joblib.dump(gb_model, os.path.join(output_dir, "model_fake_gb.pkl"))
        joblib.dump(gb_vec, os.path.join(output_dir, "vectorizer_fake_gb.pkl"))
        
        # Save ensemble
        joblib.dump(ensemble, os.path.join(output_dir, "model_fake_ensemble.pkl"))
        
        # For backward compatibility, save best model as original
        best_model = lr_model  # Logistic Regression is usually best for text
        best_vec = lr_vec
        joblib.dump(best_model, os.path.join(output_dir, "model_fake.pkl"))
        joblib.dump(best_vec, os.path.join(output_dir, "vectorizer_fake.pkl"))
        
        logger.info("✅ All models saved successfully!")
        
        results = {
            'models': {
                'logistic_regression': {'accuracy': lr_metrics['accuracy'], 'f1': lr_metrics['f1']},
                'random_forest': {'accuracy': rf_metrics['accuracy'], 'f1': rf_metrics['f1']},
                'gradient_boosting': {'accuracy': gb_metrics['accuracy'], 'f1': gb_metrics['f1']},
                'ensemble': {'accuracy': ensemble_accuracy, 'f1': None},
            },
            'best_accuracy': max(
                lr_metrics['accuracy'],
                rf_metrics['accuracy'],
                gb_metrics['accuracy'],
                ensemble_accuracy
            )
        }
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Advanced Fake News Detection Model Training")
    print("=" * 50)
    
    results = train_advanced_models()
    
    if results:
        print("\n📊 Training Results Summary:")
        print("=" * 50)
        for model_name, metrics in results['models'].items():
            print(f"{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if metrics['f1']:
                print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"\n🏆 Best Accuracy: {results['best_accuracy']:.4f}")
