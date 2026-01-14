import os
import json
import re
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import joblib
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BlogDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self.preprocess_data()
    
    def preprocess_data(self):
        return self.tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BlogClassifier:
    def __init__(self, model_path="models/blog_classifier"):
        """
        Initialize the blog classifier.
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Features that indicate personal blogs
        self.personal_features = [
            r'\bI\b', r'\bmy\b', r'\bme\b', r'\bmine\b',  # First person pronouns
            r'thoughts', r'opinion', r'experience',
            r'journey', r'learned', r'reflections',
            r'personal', r'story', r'stories',
            r'blog', r'diary', r'journal'
        ]
        
        # Features that indicate corporate content
        self.corporate_features = [
            r'buy now', r'sign up', r'free trial',
            r'limited time', r'best practices', r'top \d+',
            r'ultimate guide', r'complete guide',
            r'contact us', r'our team', r'our company',
            r'terms of service', r'privacy policy',
            'copyright', r'all rights reserved'
        ]
        
    def extract_features(self, text):
        """Extract simple features for hierarchical classification."""
        features = {}
        
        # Check for personal blog indicators
        for pattern in self.personal_features:
            count = len(re.findall(pattern, text, re.IGNORECASE))
            features[f"personal_{pattern}"] = count
            
        # Check for corporate content indicators
        for pattern in self.corporate_features:
            count = len(re.findall(pattern, text, re.IGNORECASE))
            features[f"corporate_{pattern}"] = count
            
        # Additional features
        features["text_length"] = len(text)
        features["avg_sentence_length"] = np.mean([len(s.split()) for s in re.split(r'[.!?]', text) if s.strip()]) if text else 0
        
        return features
    
    def hierarchical_classify(self, text):
        """
        Use a hierarchical approach to classification to minimize computation.
        First use simple rule-based features, then DistilBERT model if uncertain.
        """
        # Extract simple features
        features = self.extract_features(text)
        
        # Calculate personal and corporate scores
        personal_score = sum(v for k, v in features.items() if k.startswith("personal_"))
        corporate_score = sum(v for k, v in features.items() if k.startswith("corporate_"))
        
        # If there's a clear difference, make a decision
        if personal_score > corporate_score * 2:
            return "personal", personal_score / (personal_score + corporate_score + 1)
        elif corporate_score > personal_score * 2:
            return "corporate", corporate_score / (personal_score + corporate_score + 1)
        
        # If uncertain, use the DistilBERT model
        if self.model and self.tokenizer:
            try:
                # Prepare input
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get prediction
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    confidence, prediction = torch.max(probabilities, dim=1)
                    
                label = "personal" if prediction.item() == 1 else "corporate"
                return label, confidence.item()
            except Exception as e:
                logger.error(f"Error in DistilBERT prediction: {str(e)}")
        
        # Fallback to simple scoring if model fails or is not available
        total = personal_score + corporate_score + 1  # Add 1 to avoid division by zero
        return "personal" if personal_score >= corporate_score else "corporate", max(personal_score, corporate_score) / total
    
    def train(self, data_dir, test_size=0.2, batch_size=16, epochs=3, max_length=256):
        """
        Train the classifier on crawled data using DistilBERT.
        
        Args:
            data_dir (str): Directory containing crawled data
            test_size (float): Proportion of data to use for testing
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            max_length (int): Maximum sequence length for tokenization
        """
        texts = []
        labels = []
        
        try:
            # Load and prepare data
            logger.info("Loading and preparing data...")
            for filename in os.listdir(data_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Use the hierarchical classifier to generate "labels"
                        features = self.extract_features(data['content'])
                        personal_score = sum(v for k, v in features.items() if k.startswith("personal_"))
                        corporate_score = sum(v for k, v in features.items() if k.startswith("corporate_"))
                        
                        label = 1 if personal_score > corporate_score else 0  # 1 for personal, 0 for corporate
                        
                        # Truncate text to reduce memory usage
                        text = data['content'][:100000]  # Limit text length
                        texts.append(text)
                        labels.append(label)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42)
            
            # Initialize tokenizer and model
            logger.info("Initializing model and tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2
            ).to(self.device)
            
            # Create datasets and dataloaders
            train_dataset = BlogDataset(X_train, y_train, self.tokenizer, max_length=max_length)
            test_dataset = BlogDataset(X_test, y_test, self.tokenizer, max_length=max_length)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,  # Enable parallel data loading
                pin_memory=True  # Speed up data transfer to GPU
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True
            )
            
            # Training setup
            num_training_steps = len(train_loader) * epochs
            num_warmup_steps = num_training_steps // 10
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5, weight_decay=0.01)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            
            # Training loop
            best_accuracy = 0.0
            patience = 2  # Early stopping patience
            no_improve = 0
            
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                # Use tqdm for progress bar
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
                
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()
                    scheduler.step()
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f'Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}')
                
                # Evaluate
                self.model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        _, predicted = torch.max(outputs.logits, dim=1)
                        
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                accuracy = correct / total
                logger.info(f"Validation accuracy: {accuracy:.4f}")
                
                # Early stopping
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_improve = 0
                    # Save the best model
                    self.model.save_pretrained(self.model_path)
                    self.tokenizer.save_pretrained(self.model_path)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info("Early stopping triggered")
                        break
            
            return best_accuracy
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def load_model(self):
        """Load a trained model if it exists."""
        try:
            if os.path.exists(self.model_path):
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path).to(self.device)
                logger.info(f"Model loaded from {self.model_path}")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    classifier = BlogClassifier()
    
    # Train if data is available
    if os.path.exists("data"):
        classifier.train("data")
    
    # Test classification
    test_texts = [
        "I recently started learning about machine learning and wanted to share my experience. It's been a fascinating journey so far.",
        "Our company provides the best SEO services in the industry. Sign up for a free consultation today!"
    ]
    
    for text in test_texts:
        label, confidence = classifier.hierarchical_classify(text)
        print(f"Text: {text[:50]}...")
        print(f"Classification: {label} (confidence: {confidence:.4f})")
        print("-" * 50) 