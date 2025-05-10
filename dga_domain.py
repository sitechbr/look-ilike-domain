import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import re

class DGADetector:
    def __init__(self, threshold=0.8):
        """
        Initialize the DGA detector with a confidence threshold.
        
        Args:
            threshold (float): Confidence threshold (0-1) below which a domain is considered DGA.
                              Default is 0.8.
        """
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self.model = None  # Will be initialized after seeing data
    
    def _build_model(self, input_dim):
        """Build the neural network model architecture."""
        model = Sequential([
            Input(shape=(input_dim,)),  # Explicit input layer
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_data(self, domains, labels):
        """
        Preprocess the domain data and split into train/test sets.
        
        Args:
            domains (list): List of domain names
            labels (list): Corresponding labels (1 for legitimate, 0 for DGA)
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Extract domain parts (without TLD)
        cleaned_domains = [self._clean_domain(d) for d in domains]
        
        # Vectorize the domains
        X = self.vectorizer.fit_transform(cleaned_domains).toarray()
        y = np.array(labels)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def _clean_domain(self, domain):
        """Clean the domain by removing protocol and TLD."""
        domain = re.sub(r'^https?://', '', domain)  # Remove http(s)://
        domain = re.sub(r'^www\.', '', domain)      # Remove www.
        domain = re.sub(r'\..*$', '', domain)       # Remove TLD
        return domain
    
    def train(self, domains, labels, epochs=20, batch_size=64):
        """
        Train the DGA detection model.
        
        Args:
            domains (list): List of domain names
            labels (list): Corresponding labels (1 for legitimate, 0 for DGA)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        # First fit the vectorizer and get the actual feature count
        cleaned_domains = [self._clean_domain(d) for d in domains]
        X = self.vectorizer.fit_transform(cleaned_domains)
        n_features = X.shape[1]
        
        # Build model with correct input dimension
        self.model = self._build_model(n_features)
        
        # Convert to array and split
        X = X.toarray()
        y = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate on test set
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel trained with test accuracy: {accuracy:.1%}")
    
    def predict(self, domain):
        """
        Predict whether a domain is legitimate or DGA-generated.
        
        Args:
            domain (str): Domain name to evaluate
            
        Returns:
            tuple: (score, is_dga)
                   score: confidence score (0-1) that domain is legitimate
                   is_dga: boolean indicating if domain is likely DGA-generated
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        cleaned_domain = self._clean_domain(domain)
        X = self.vectorizer.transform([cleaned_domain]).toarray()
        score = self.model.predict(X, verbose=0)[0][0]
        return score, score < self.threshold
    
    def evaluate(self, domains, labels):
        """
        Evaluate the model on a test set and return accuracy.
        
        Args:
            domains (list): List of domain names
            labels (list): Corresponding labels (1 for legitimate, 0 for DGA)
            
        Returns:
            float: Accuracy score
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        cleaned_domains = [self._clean_domain(d) for d in domains]
        X = self.vectorizer.transform(cleaned_domains).toarray()
        y = np.array(labels)
        
        y_pred = (self.model.predict(X) > 0.5).astype(int)
        return accuracy_score(y, y_pred)


# Example usage
if __name__ == "__main__":
    # Example data - in practice you would use a much larger dataset
    legitimate_domains = [
        "google.com", "youtube.com", "facebook.com", "amazon.com", 
        "wikipedia.org", "reddit.com", "instagram.com", "linkedin.com",
        "netflix.com", "microsoft.com", "apple.com", "twitter.com"
    ]
    
    dga_domains = [
        "123123.com", "xyz789.net", "456abc.org", "qwerty123.info",
        "asdfg456.biz", "zxcvb789.xyz", "098765.co", "poiuyt.net",
        "lkjhgf.org", "mnbvcx.info", "987654.biz", "543210.xyz"
    ]
    
    # Prepare full dataset
    domains = legitimate_domains + dga_domains
    labels = [1] * len(legitimate_domains) + [0] * len(dga_domains)
    
    # Initialize and train the detector
    detector = DGADetector(threshold=0.8)
    detector.train(domains, labels, epochs=15)
    
    # Test some domains
    test_domains = [
        "google.com",        # legitimate
        "bankofamerica.com", # legitimate
        "1234567890.net",    # likely DGA
        "rand0mstr1ng.xyz",  # likely DGA
        "microsoft.com",     # legitimate
        "asdf1234.info"     # likely DGA
    ]
    
    print("\nTesting domains:")
    for domain in test_domains:
        score, is_dga = detector.predict(domain)
        print(f"{domain}: Score = {score:.4f} | {'DGA' if is_dga else 'Legitimate'}")