def main():
    processor = TenderDocumentProcessor()
    
    # Extract text from PDFs
    document_path = "path/to/tender.pdf"
    extracted_text = processor.extract_text_from_pdf(document_path)
    
    # Prepare training data
    training_documents = [
        {
            "text": "extracted_text",
            "labels": ["scope_of_work", "pricing", "deadlines"]
        }
        # Add more documents...
    ]
    
    training_config = processor.prepare_training_data(training_documents)
    
    # Train the classifier
    classifier_arn = processor.train_classifier(training_config)

if __name__ == "__main__":
    main()
