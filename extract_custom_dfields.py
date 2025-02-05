def extract_custom_fields(self, text):
    """Add custom extraction rules"""
    patterns = {
        'tender_id': r'Tender\s+No\.?\s*:?\s*([A-Z0-9-/]+)',
        'submission_deadline': r'Submission\s+Deadline\s*:?\s*([^\n]+)',
        'estimated_value': r'Estimated\s+Value\s*:?\s*([^\n]+)'
    }
    
    results = {}
    for field, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        results[field] = matches
    
    return results
