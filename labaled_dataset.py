# Example of creating training data
training_texts = []  # List of preprocessed tender documents
labels = []         # Corresponding labels

# Add your labeled data
for document, label in your_labeled_data:
    processed_text = processor.preprocess_text(document)
    training_texts.append(processed_text)
    labels.append(label)

# Train the model
processor.train_model(training_texts, labels)
