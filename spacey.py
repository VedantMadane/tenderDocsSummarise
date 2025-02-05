import pytesseract
from pdf2image import convert_from_path
import spacy
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class TenderDocumentProcessor:
    def __init__(self):
        # Load spaCy model for NLP tasks
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier()

    def extract_text_from_pdf(self, pdf_path):
        """Convert PDF to text using pdf2image and Tesseract OCR"""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            
            # Extract text from each page
            text_content = []
            for image in images:
                text = pytesseract.image_to_string(image)
                text_content.append(text)
            
            return '\n'.join(text_content)
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None

    def preprocess_text(self, text):
        """Clean and preprocess extracted text"""
        doc = self.nlp(text)
        
        # Basic preprocessing
        tokens = [token.text.lower() for token in doc
                 if not token.is_stop and not token.is_punct
                 and token.text.strip()]
        
        return ' '.join(tokens)

    def train_model(self, training_data, labels):
        """Train a document classifier"""
        # Convert text to features
        X = self.vectorizer.fit_transform(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Print accuracy
        score = self.classifier.score(X_test, y_test)
        print(f"Model accuracy: {score:.2f}")

    def extract_information(self, text):
        """Extract specific information using rules and NLP"""
        doc = self.nlp(text)
        
        # Dictionary to store extracted information
        extracted_info = {
            'dates': [],
            'amounts': [],
            'organizations': [],
            'key_terms': []
        }
        
        # Extract dates and amounts
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                extracted_info['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':
                extracted_info['amounts'].append(ent.text)
            elif ent.label_ == 'ORG':
                extracted_info['organizations'].append(ent.text)
        
        return extracted_info

def main():
    # Initialize processor
    processor = TenderDocumentProcessor()
    
    # Example usage
    pdf_path = "D:/Projects/devanagari-text-extraction/data/raw/2.pdf"
    
    # Extract text from PDF
    text = processor.extract_text_from_pdf(pdf_path)
    if text:
        # Preprocess text
        processed_text = processor.preprocess_text(text)
        
        # Extract information
        information = processor.extract_information(text)
        
        # Print results
        print("\nExtracted Information:")
        for key, values in information.items():
            print(f"\n{key.capitalize()}:")
            for value in set(values):  # Remove duplicates
                print(f"- {value}")

if __name__ == "__main__":
    main()
