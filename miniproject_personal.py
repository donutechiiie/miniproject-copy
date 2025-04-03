
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.linear_model import LogisticRegression
import time
import re

app = Flask(__name__)
from flask_cors import CORS
CORS(app)  # Enable CORS for React frontend

# Supabase credentials
url = "https://jujxoskixfadyvrxlaru.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1anhvc2tpeGZhZHl2cnhsYXJ1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjM4MjI5NSwiZXhwIjoyMDU3OTU4Mjk1fQ.Ka4RjOSpSr5ODKpklgvFJkx9iNPxgqwIMFLbQU5-NMo"
supabase: Client = create_client(url, key)

# Base nutritional targets
base_targets = {
    'breakfast': {
        'carbohydrate': 90, 'protein': 30, 'total_fat': 24, 'dietary_fibre_total': 10, 'total_ascorbic_acid': 20,
        'calcium_mg': 300, 'iron_mg': 4, 'vite': 5, 'linoleic_c18_2n6': 5, 'alpha_linolenic_c18_3n3': 0.5,
        'lys': 1.5, 'met': 0.5, 'total_folates': 100, 'beta_carotene': 2000, 'magnesium_mg': 100, 'zinc_mg': 3,
        'total_free_sugars': 20, 'total_polyphenols': 50
    },
    'lunch': {
        'carbohydrate': 120, 'protein': 40, 'total_fat': 32, 'dietary_fibre_total': 15, 'total_ascorbic_acid': 30,
        'calcium_mg': 400, 'iron_mg': 6, 'vite': 7, 'linoleic_c18_2n6': 7, 'alpha_linolenic_c18_3n3': 0.7,
        'lys': 2.0, 'met': 0.7, 'total_folates': 150, 'beta_carotene': 3000, 'magnesium_mg': 150, 'zinc_mg': 4,
        'total_free_sugars': 25, 'total_polyphenols': 75
    },
    'dinner': {
        'carbohydrate': 90, 'protein': 30, 'total_fat': 24, 'dietary_fibre_total': 10, 'total_ascorbic_acid': 20,
        'calcium_mg': 300, 'iron_mg': 4, 'vite': 5, 'linoleic_c18_2n6': 5, 'alpha_linolenic_c18_3n3': 0.5,
        'lys': 1.5, 'met': 0.5, 'total_folates': 100, 'beta_carotene': 2000, 'magnesium_mg': 100, 'zinc_mg': 3,
        'total_free_sugars': 20, 'total_polyphenols': 50
    }
}

# Health condition adjustments
health_adjustments = {
    'high_glucose': {'carbohydrate': 0.6, 'total_free_sugars': 0.4, 'dietary_fibre_total': 1.3, 'total_polyphenols': 1.2},
    'low_iron': {'iron_mg': 1.5, 'total_ascorbic_acid': 1.2},
    'high_cholesterol': {'total_fat': 0.5, 'linoleic_c18_2n6': 1.3, 'dietary_fibre_total': 1.2},
    'low_vitamin_a': {'beta_carotene': 1.5},
    'low_folate': {'total_folates': 1.5},
    'low_magnesium': {'magnesium_mg': 1.5},
    'low_zinc': {'zinc_mg': 1.5},
    'high_blood_pressure': {'magnesium_mg': 1.2, 'calcium_mg': 1.2, 'total_fat': 0.8},
    'low_hemoglobin': {'iron_mg': 1.5, 'total_folates': 1.5, 'total_ascorbic_acid': 1.2}
}

# Fetch and preprocess food data at startup
print("Fetching and merging data...")
start_time = time.time()
tables = {
    "nutritionaldata": "nutritional_df", "watersolublevitamins": "vitamins_df", "amino_acid_profile": "amino_df",
    "carotenoids": "carotenoids_df", "edibleoils": "oils_df", "fat_soluble_vitamins": "fat_vitamins_df",
    "fatty_acid_profile": "fatty_acids_df", "minerals_trace_elements": "minerals_df",
    "oligosaccharides_phytosterols": "oligo_df", "organic_acids": "organic_df", "polyphenols": "polyphenols_df",
    "starch_and_sugars": "starch_df"
}
dataframes = {}
for table, df_name in tables.items():
    try:
        response = supabase.table(table).select("*").execute()
        dataframes[df_name] = pd.DataFrame(response.data)
    except Exception as e:
        print(f"Error fetching {table}: {e}")
        dataframes[df_name] = pd.DataFrame()

combined_df = dataframes["nutritional_df"]
for df_name, df in dataframes.items():
    if df_name != "nutritional_df" and not df.empty:
        combined_df = combined_df.merge(df, on='food_code', how='left', suffixes=('', f'_{df_name}'))
combined_df.fillna(0, inplace=True)
print(f"Data fetched and merged in {time.time() - start_time:.2f} seconds, Shape: {combined_df.shape}")

# Fetch user data and blood reports
def get_user_data(user_id):
    try:
        users_response = supabase.table("UserTable").select("auth_uid, notes").eq("auth_UID", user_id).execute()
        reports_response = supabase.table("reports").select("auth_uid, extracted_text").eq("auth_UID", user_id).execute()
    except Exception as e:
        print(f"Error fetching user data for {user_id}: {e}")
        return {'restrictions': [], 'reports': []}
    
    user_data = {'restrictions': [], 'reports': []}
    if users_response.data:
        notes = users_response.data[0].get('notes') or {}
        restrictions = notes.get('restrictions', []) if isinstance(notes, dict) else []
        user_data['restrictions'] = [r.lower() for r in restrictions if isinstance(r, str)]
    for report in reports_response.data:
        if report['extracted_text']:
            user_data['reports'].append(report['extracted_text'])
    return user_data

# Parse blood report text with hemoglobin
def parse_blood_report(text):
    conditions = {}
    patterns = {
        'glucose': (r'glucose\D*(\d+\.?\d*)', 130, 'high_glucose', True),
        'iron': (r'iron\D*(\d+\.?\d*)', 12, 'low_iron', False),
        'cholesterol': (r'cholesterol\D*(\d+\.?\d*)', 200, 'high_cholesterol', True),
        'folate': (r'folate\D*(\d+\.?\d*)', 400, 'low_folate', False),
        'magnesium': (r'magnesium\D*(\d+\.?\d*)', 1.7, 'low_magnesium', False),
        'zinc': (r'zinc\D*(\d+\.?\d*)', 0.7, 'low_zinc', False),
        'vitamin a': (r'vitamin a\D*(\d+\.?\d*)', 20, 'low_vitamin_a', False),
        'blood pressure': (r'blood pressure\D*(\d+/\d+)', '140/90', 'high_blood_pressure', True),
        'hemoglobin': (r'haemoglobin\D*(\d+\.?\d*)\s*g/dl', 12.5, 'low_hemoglobin', False)
    }
    for key, (pattern, threshold, condition, is_high) in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key == 'blood pressure':
                sys, dia = map(int, value.split('/'))
                if sys > int(threshold.split('/')[0]) or dia > int(threshold.split('/')[1]):
                    conditions[condition] = True
            else:
                value = float(value)
                if (is_high and value > threshold) or (not is_high and value < threshold):
                    conditions[condition] = True
    return conditions

# Adjust targets based on health conditions
def adjust_targets(base_targets, conditions):
    adjusted = {meal: targets.copy() for meal, targets in base_targets.items()}
    for meal_type in adjusted:
        for condition, adjustments in health_adjustments.items():
            if conditions.get(condition, False):
                for nutrient, factor in adjustments.items():
                    adjusted[meal_type][nutrient] = adjusted[meal_type].get(nutrient, 0) * factor
    return adjusted

# Train logistic regression with relaxed criteria
def train_logistic_regression(df, meal_type, targets):
    features = list(targets[meal_type].keys())
    X = df[features].values
    y = np.array([
        1 if sum(df.iloc[i][n] >= 0 and df.iloc[i][n] <= targets[meal_type][n] * 1.2 for n in features) >= len(features) * 0.7
        else 0 for i in range(len(df))
    ])
    if sum(y) == 0:
        print(f"Warning: No foods meet criteria for {meal_type}. Using fallback labeling.")
        y = np.array([
            1 if sum(df.iloc[i][n] <= targets[meal_type][n] * 1.5 for n in features) >= len(features) * 0.5
            else 0 for i in range(len(df))
        ])
    if sum(y) == 0:
        print(f"Warning: Still no suitable foods for {meal_type}. Using extreme fallback.")
        scores = np.array([
            sum(abs(df.iloc[i][n] - targets[meal_type][n]) / targets[meal_type][n] for n in features)
            for i in range(len(df))
        ])
        threshold = np.percentile(scores, 10)
        y = np.array([1 if scores[i] <= threshold else 0 for i in range(len(df))])
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X, y)
    return clf

# Recommend personalized meals
def recommend_meals_lr(meal_type, df, restrictions, targets, num_meals=3):
    if restrictions:
        pattern = '|'.join([fr'\b{restr}\b' for restr in restrictions] + [fr'{restr}.*' for restr in restrictions])
        filtered_df = df[~df['food_name_nutri'].str.lower().str.contains(pattern, na=False, case=False)]
    else:
        filtered_df = df.copy()

    clf = train_logistic_regression(filtered_df, meal_type, targets)
    features = list(targets[meal_type].keys())
    X = filtered_df[features].values
    predictions = clf.predict(X)
    suitable_foods = filtered_df[predictions == 1].sample(frac=1, random_state=42).reset_index(drop=True)

    if len(suitable_foods) == 0:
        return [{'meal': [], 'totals': {k: 0 for k in targets[meal_type]}}] * num_meals

    meal_options = []
    used_meal_sets = set()
    min_targets = {k: v * 0.7 for k, v in targets[meal_type].items()}
    max_targets = {k: v * 1.2 for k, v in targets[meal_type].items()}

    for meal_num in range(num_meals):
        meal = []
        totals = {key: 0 for key in targets[meal_type].keys()}
        shuffled_foods = suitable_foods.sample(frac=1, random_state=meal_num * 42).reset_index(drop=True)
        idx = 0

        while idx < len(shuffled_foods) and not all(totals[n] >= min_targets[n] for n in totals):
            row = shuffled_foods.iloc[idx]
            food = row.get('food_name_nutri') or row.get('food_name')
            temp_totals = totals.copy()
            for nutrient in totals:
                temp_totals[nutrient] += row.get(nutrient, 0)
            if all(temp_totals[n] <= max_targets[n] for n in temp_totals):
                meal.append(food)
                totals = temp_totals
            idx += 1

        meal_tuple = tuple(sorted(meal))
        if meal_tuple and meal_tuple not in used_meal_sets:
            used_meal_sets.add(meal_tuple)
            meal_options.append({'meal': meal, 'totals': totals})

        if len(meal_options) <= meal_num and idx >= len(shuffled_foods):
            meal = suitable_foods['food_name_nutri'].iloc[:min(3, len(suitable_foods))].tolist()
            totals = suitable_foods[features].iloc[:min(3, len(suitable_foods))].sum().to_dict()
            meal_tuple = tuple(sorted(meal))
            if meal_tuple not in used_meal_sets:
                used_meal_sets.add(meal_tuple)
                meal_options.append({'meal': meal, 'totals': totals})

    while len(meal_options) < num_meals:
        meal = meal_options[0]['meal'][:2] + [suitable_foods['food_name_nutri'].iloc[min(len(meal_options), len(suitable_foods)-1)]]
        totals = suitable_foods[features].iloc[:3].sum().to_dict()
        meal_tuple = tuple(sorted(meal))
        if meal_tuple not in used_meal_sets:
            used_meal_sets.add(meal_tuple)
            meal_options.append({'meal': meal, 'totals': totals})

    return meal_options[:num_meals]

# API Endpoint to trigger personalized recommendations
@app.route('/api/generate-personalized-recommendations', methods=['POST'])
def generate_personalized_recommendations():
    data = request.get_json()
    user_id = data.get('user_id')
    num_meals = data.get('num_meals', 3)

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    start_time = time.time()
    user_data = get_user_data(user_id)
    restrictions = user_data['restrictions']
    reports = user_data['reports']
    conditions = {}
    for report in reports:
        conditions.update(parse_blood_report(report))

    user_targets = adjust_targets(base_targets, conditions)
    all_recommendations = {}

    for meal_type in ['breakfast', 'lunch', 'dinner']:
        options = recommend_meals_lr(meal_type, combined_df, restrictions, user_targets, num_meals)
        all_recommendations[meal_type] = [
            {
                'option': i + 1,
                'ingredients': option['meal'],
                'nutritional_totals': {k: round(float(v), 2) for k, v in option['totals'].items()}
            } for i, option in enumerate(options)
        ]

    response = {
        'user_id': user_id,
        'recommendations': all_recommendations,
        'execution_time': f"{time.time() - start_time:.2f} seconds"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Runs on port 5001 to avoid conflict with standard1