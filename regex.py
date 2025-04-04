import json

# Define relevant test markers for filtering
RELEVANT_MARKERS = {
    "Blood Sugar Test": ["Fasting Blood Sugar", "Post-Meal Blood Sugar", "Postprandial Blood Sugar (PPBS)", "HbA1c"],
    "Lipid Profile": ["Total Cholesterol", "HDL","HDL- Cholesterol","LDL- Cholestrol" "LDL", "Triglycerides", "Serum Cholesterol", "HDL-Cholesterol"],
    "Kidney Function Test": ["Serum Creatinine", "Blood Urea Nitrogen", "Uric Acid", "eGFR"],
    "Liver Function Test": ["SGPT", "SGOT", "Total Bilirubin", "Albumin", "Alkaline Phosphatase"],
    "Thyroid Function Test": ["T3", "T4", "TSH"],
    "Hemogram / CBC": ["Haemoglobin", "RBC", "WBC", "Platelet Count","Mean Platelet Count"],
    "Electrolytes Test": ["Serum Sodium (Na+)","S.Potassium (k+) ","Sodium", "Potassium", "Chloride", "Calcium"],
    "Vitamin Tests": ["Vitamin D", "Vitamin B12"]
}

# Load the extracted data JSON file
input_file = "dataset/extracted_text/lab_reports_sample.json"
output_file = "dataset/extracted_text/filtered_reports_new.json"

with open(input_file, "r", encoding="utf-8") as file:
    extracted_data = json.load(file)

# Filter only relevant markers
filtered_results = {}

for filename, tests in extracted_data.items():
    filtered_tests = []

    for test in tests:
        test_name = test.get("Test Name", "").strip()

        # Check if the test name matches any relevant marker (including new ones)
        if any(test_name in markers for markers in RELEVANT_MARKERS.values()):
            filtered_tests.append(test)

    # Store results only if relevant markers are found
    if filtered_tests:
        filtered_results[filename] = filtered_tests

# Save the filtered results to a new JSON file
with open(output_file, "w", encoding="utf-8") as output:
    json.dump(filtered_results, output, indent=4)

print(f"✅ Filtered data saved to '{output_file}'")