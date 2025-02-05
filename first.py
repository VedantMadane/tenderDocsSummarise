import boto3
import json
from typing import List, Dict

class TenderDocumentProcessor:
    def __init__(self):
        self.comprehend = boto3.client('comprehend')
        self.textract = boto3.client('textract')
        self.s3 = boto3.client('s3')

    def prepare_training_data(self, documents: List[Dict]):
        """
        Prepare labeled training data for the document classifier
        documents: List of dictionaries containing document text and labels
        """
        training_data = {
            'DocumentClassifierName': 'TenderDocumentClassifier',
            'DataAccessRoleArn': os.environ['DATA_ACCESS_ROLE_ARN'],
            'InputDataConfig': {
                'S3Uri': os.environ['TRAINING_DATA_S3_URI'],
                'LabelDelimiter': '|'
            },
            'LanguageCode': 'en'
        }
        return training_data

    def extract_text_from_pdf(self, document_path: str):
        """
        Extract text from PDF using Amazon Textract
        """
        with open(document_path, 'rb') as document:
            bytes_document = document.read()
        
        response = self.textract.analyze_document(
            Document={'Bytes': bytes_document},
            FeatureTypes=['FORMS', 'TABLES']
        )
        return response

    def train_classifier(self, training_data: Dict):
        """
        Train the document classifier using Amazon Comprehend
        """
        response = self.comprehend.create_document_classifier(**training_data)
        return response['DocumentClassifierArn']
