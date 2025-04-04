import cv2
import pytesseract
import os
import json
import re

def extract_text_from_image(image_path):
    """Extracts text from a given image using Tesseract OCR."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def clean_text(text):
    """Removes unnecessary spaces and fixes OCR errors."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_lab_report(text):
    """Parses lab report text into structured JSON format."""
    lines = text.split('\n')
    data = []
    
    for i in range(len(lines)):
        line = clean_text(lines[i])
        
        # Match patterns like "Haemoglobin 15.8 g/dl 13.0-16.5 g/dl"
        match = re.match(r'^(.*?)\s+([0-9.,]+\s*\S+)\s+([0-9.,-]+\s*\S+)$', line)
        
        if match:
            test_name = match.group(1)
            result = match.group(2)
            reference_range = match.group(3)
            
            data.append({
                "Test Name": test_name,
                "Result": result,
                "Reference Range": reference_range
            })
    
    return data

def process_reports(directory):
    """Processes all images in the processed_reports folder and extracts structured data."""
    all_reports = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(directory, filename)
            extracted_text = extract_text_from_image(image_path)
            structured_data = parse_lab_report(extracted_text)
            
            all_reports[filename] = structured_data
    
    return all_reports

# Directory containing processed reports
reports_dir = "dataset/processed_reports"
extracted_reports = process_reports(reports_dir)

# Save extracted data as JSON
output_path = "dataset/extracted_text/lab_reports.json"
with open(output_path, "w") as f:
    json.dump(extracted_reports, f, indent=4)

print(f"Extracted data saved to {output_path}")