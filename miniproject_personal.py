from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.tree import DecisionTreeClassifier
import time
import random
import json
from flask_cors import CORS
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Supabase configuration
SUPABASE_URL = "https://jujxoskixfadyvrxlaru.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1anhvc2tpeGZhZHl2cnhsYXJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIzODIyOTUsImV4cCI6MjA1Nzk1ODI5NX0.WcDFgHAMGbsksGUto44U4ke33yz_hONETzQX3U6-VcQ"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Nutritional requirements and personalization factors
DAILY_NUTRIENTS = {
    'carbohydrate': 300,  # grams (normal baseline)
    'protein': 50,        # grams
    'total_fat': 70,      # grams
    'iron_mg': 18,        # mg
}
PERSONALIZED_ADJUSTMENTS = {
    'high_sugar': {'carbohydrate': 0.7},  # Reduce carbs by 30% for high sugar
    'high_cholesterol': {'total_fat': 0.7},  # Reduce fat by 30% for high cholesterol
}

# Normal ranges (aligned with UserTable's gender field)
NORMAL_RANGES = {
    'cholesterol': {'min': 125, 'max': 200, 'unit': 'mg/dl'},
    'hemoglobin': {'male': {'min': 13.5, 'max': 17.5, 'unit': 'g/dl'},
                   'female': {'min': 12.0, 'max': 15.5, 'unit': 'g/dl'}},
    'sugar': {'min': 70, 'max': 99, 'unit': 'mg/dl'}
}

MEAL_TYPES = ['breakfast', 'lunch', 'dinner']
MEALS_PER_TYPE = 3  # Fixed to 3 meals per type
used_meal_names = set()  # To ensure unique meal names
user_meal_combinations = {}  # Dictionary to track combinations per user
all_recommendations = []  # Store all recommendations globally

# Fetch and preprocess data once at startup
print("Loading data...")
start_time = time.time()
tables = ['nutritionaldata', 'minerals_trace_elements']
dataframes = {}
for table in tables:
    response = supabase.table(table).select("*").execute()
    df_name = f"{table}_df"
    dataframes[df_name] = pd.DataFrame(response.data)

# Convert preloaded data to dictionary format
food_data = {}
nutri_df = dataframes['nutritionaldata_df']
for _, row in nutri_df.iterrows():
    food_code = row['food_code']
    food_data[food_code] = {
        'food_name': row['food_name'] or f"Unnamed_{food_code}",
        'carbohydrate': row.get('carbohydrate', None),
        'protein': row.get('protein', None),
        'total_fat': row.get('total_fat', None),
        'iron_mg': None
    }

minerals_df = dataframes['minerals_trace_elements_df']
for _, row in minerals_df.iterrows():
    food_code = row['food_code']
    if food_code in food_data:
        food_data[food_code].update({'iron_mg': row.get('iron_mg', None)})
    else:
        food_data[food_code] = {
            'food_name': row['food_name'] or f"Unnamed_{food_code}",
            'carbohydrate': None,
            'protein': None,
            'total_fat': None,
            'iron_mg': row.get('iron_mg', None)
        }

print(f"Data loaded in {time.time() - start_time:.2f} seconds. Loaded {len(food_data)} food items.")

# Fetch data for a specific user
def get_user_data(user_id):
    try:
        response = supabase.table('UserTable').select('auth_uid, notes, gender').eq('auth_uid', user_id).execute()
        if response.data:
            row = response.data[0]
            return {
                'restrictions': row['notes'] if row['notes'] else [],
                'gender': row['gender'] if row['gender'] else 'unknown'
            }
        return {'restrictions': [], 'gender': 'unknown'}
    except Exception as e:
        print(f"Error fetching user data for {user_id}: {e}")
        return {'restrictions': [], 'gender': 'unknown'}

# Fetch and parse the latest report for a user
def get_user_report(user_id):
    try:
        response = supabase.table('reports').select('extracted_text').eq('auth_uid', user_id).order('created_at', desc=True).limit(1).execute()
        if response.data:
            text = response.data[0]['extracted_text']
            if text:
                cholesterol = float(re.search(r'cholesterol (\d+\.?\d*) mg/dl', text, re.IGNORECASE).group(1)) if re.search(r'cholesterol (\d+\.?\d*) mg/dl', text, re.IGNORECASE) else None
                sugar = float(re.search(r'sugar (\d+\.?\d*) mg/dl', text, re.IGNORECASE).group(1)) if re.search(r'sugar (\d+\.?\d*) mg/dl', text, re.IGNORECASE) else None
                hemoglobin = float(re.search(r'hemoglobin (\d+\.?\d*) g/dl', text, re.IGNORECASE).group(1)) if re.search(r'hemoglobin (\d+\.?\d*) g/dl', text, re.IGNORECASE) else None
                return {'cholesterol': cholesterol, 'sugar': sugar, 'hemoglobin': hemoglobin}
    except Exception as e:
        print(f"Error fetching report for user {user_id}: {e}")
    return None

# Determine health status based on report and gender
def get_health_status(report, gender):
    status = {}
    if report and report['cholesterol'] is not None:
        status['cholesterol'] = 'high' if report['cholesterol'] > NORMAL_RANGES['cholesterol']['max'] else 'normal'
    if report and report['sugar'] is not None:
        status['sugar'] = 'high' if report['sugar'] > NORMAL_RANGES['sugar']['max'] else 'normal'
    if report and report['hemoglobin'] is not None:
        hb_range = NORMAL_RANGES['hemoglobin']['male' if gender.lower() == 'male' else 'female']
        status['hemoglobin'] = 'low' if report['hemoglobin'] < hb_range['min'] else 'high' if report['hemoglobin'] > hb_range['max'] else 'normal'
    return status

# Adjust daily nutrients based on health status
def adjust_nutrients(base_nutrients, health_status):
    adjusted = base_nutrients.copy()
    for condition, factors in PERSONALIZED_ADJUSTMENTS.items():
        if health_status.get('sugar') == 'high' and condition == 'high_sugar':
            adjusted['carbohydrate'] = int(base_nutrients['carbohydrate'] * factors['carbohydrate'])
        if health_status.get('cholesterol') == 'high' and condition == 'high_cholesterol':
            adjusted['total_fat'] = int(base_nutrients['total_fat'] * factors['total_fat'])
    return adjusted

# Decision tree to classify if a meal meets nutritional balance
def train_decision_tree(food_data, daily_nutrients):
    X = []
    y = []
    feature_names = ['carbohydrate', 'protein', 'total_fat', 'iron_mg']
    
    for food_code, nutrients in food_data.items():
        features = [nutrients.get(feat, 0) if nutrients.get(feat) is not None else 0 for feat in feature_names]
        X.append(features)
        balanced_count = sum(1 for feat, val in zip(feature_names, features) if val > 0 and daily_nutrients[feat] * 0.1 <= val <= daily_nutrients[feat] * 0.5)
        y.append(1 if balanced_count >= 2 else 0)
    
    if not X:
        print("ERROR: No training data for decision tree.")
        return None, feature_names
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    return clf, feature_names

# Generate a unique meal
def generate_meal(food_data, clf, feature_names, used_ingredients, restricted_foods, daily_nutrients):
    available_foods = {
        k: v for k, v in food_data.items() 
        if k not in used_ingredients and v['food_name'].lower() not in [r.lower() for r in restricted_foods]
    }
    if len(available_foods) < 2:
        return None
    
    meal_ingredients = random.sample(list(available_foods.keys()), min(3, len(available_foods)))
    total_nutrients = {}
    
    for food_code in meal_ingredients:
        nutrients = food_data[food_code]
        for key, value in nutrients.items():
            if key != 'food_name' and isinstance(value, (int, float)) and value is not None:
                total_nutrients[key] = total_nutrients.get(key, 0) + value
    
    X_test = [total_nutrients.get(feat, 0) for feat in feature_names]
    is_balanced = clf.predict([X_test])[0] if clf else True
    
    if not is_balanced:
        return None
    
    base_name = random.choice(['Mix', 'Combo', 'Plate'])
    ingredient_name = food_data[meal_ingredients[0]]['food_name'].split()[0]
    meal_name = f"{ingredient_name} {base_name} {random.randint(1, 100)}"
    while meal_name in used_meal_names:
        meal_name = f"{ingredient_name} {base_name} {random.randint(1, 100)}"
    used_meal_names.add(meal_name)
    
    return {
        'meal_name': meal_name,
        'meal_ingredients': [food_data[code]['food_name'] for code in meal_ingredients],
        'nutrition_total': {
            'carbohydrate': total_nutrients.get('carbohydrate', 0),
            'protein': total_nutrients.get('protein', 0),
            'total_fat': total_nutrients.get('total_fat', 0),
            'iron_mg': total_nutrients.get('iron_mg', 0)
        },
        'ingredient_codes': meal_ingredients
    }

# Delete existing recommendations for a user and meal type
def delete_existing_recommendations(user_id, meal_type):
    try:
        response = supabase.table('personal_recommendation').delete().eq('user_id', user_id).eq('meal_type', meal_type).execute()
        print(f"Deleted existing recommendations for user {user_id} and meal type {meal_type}.")
    except Exception as e:
        print(f"Error deleting existing recommendations for user {user_id} and meal type {meal_type}: {e}")

# Recommend meals for a user
def recommend_meals_for_user(user_id, food_data, clf, feature_names, restricted_foods, daily_nutrients):
    if user_id not in user_meal_combinations:
        user_meal_combinations[user_id] = set()
    
    used_combinations = user_meal_combinations[user_id]
    recommendations = []
    
    for meal_type in MEAL_TYPES:
        num_meals = MEALS_PER_TYPE
        print(f"\n{meal_type.capitalize()} Meals:")
        attempts = 0
        max_attempts = 10
        
        for i in range(num_meals):
            meal_generated = False
            while attempts < max_attempts:
                meal = generate_meal(food_data, clf, feature_names, set(), restricted_foods, daily_nutrients)
                if meal:
                    combination_key = tuple(sorted(meal['ingredient_codes']))
                    if combination_key not in used_combinations:
                        rec = {
                            'user_id': user_id,
                            'meal_type': meal_type,
                            'meal_name': meal['meal_name'],
                            'meal_ingredients': ', '.join(meal['meal_ingredients']),
                            'nutrition_total': json.dumps(meal['nutrition_total']),
                            'created_at': datetime.now().isoformat()
                        }
                        recommendations.append(rec)
                        all_recommendations.append(rec)
                        used_combinations.add(combination_key)
                        print(f"User ID: {user_id}")
                        print(f"Meal Type: {meal_type}")
                        print(f"Meal Name: {meal['meal_name']}")
                        print("Ingredients:")
                        for ingredient in meal['meal_ingredients']:
                            print(f"  - {ingredient}")
                        print("Nutritional Value:")
                        nutrients = meal['nutrition_total']
                        print(f"  - Carbohydrates: {nutrients['carbohydrate']} g")
                        print(f"  - Proteins: {nutrients['protein']} g")
                        print(f"  - Fats: {nutrients['total_fat']} g")
                        print(f"  - Iron: {nutrients['iron_mg']} mg")
                        print()
                        meal_generated = True
                        break
                attempts += 1
            
            if not meal_generated and all_recommendations:
                fallback_rec = random.choice(all_recommendations)
                fallback_meal = {
                    'meal_name': fallback_rec['meal_name'],
                    'meal_ingredients': fallback_rec['meal_ingredients'].split(', '),
                    'nutrition_total': json.loads(fallback_rec['nutrition_total']),
                    'ingredient_codes': [code for code in food_data if food_data[code]['food_name'] in fallback_rec['meal_ingredients'].split(', ')]
                }
                if not any(ingredient.lower() in [r.lower() for r in restricted_foods] for ingredient in fallback_meal['meal_ingredients']):
                    combination_key = tuple(sorted(fallback_meal['ingredient_codes']))
                    if combination_key not in used_combinations:
                        rec = {
                            'user_id': user_id,
                            'meal_type': meal_type,
                            'meal_name': fallback_meal['meal_name'],
                            'meal_ingredients': ', '.join(fallback_meal['meal_ingredients']),
                            'nutrition_total': json.dumps(fallback_meal['nutrition_total']),
                            'created_at': datetime.now().isoformat()
                        }
                        recommendations.append(rec)
                        used_combinations.add(combination_key)
                        print(f"User ID: {user_id}")
                        print(f"Meal Type: {meal_type}")
                        print(f"Meal Name: {fallback_meal['meal_name']}")
                        print("Ingredients:")
                        for ingredient in fallback_meal['meal_ingredients']:
                            print(f"  - {ingredient}")
                        print("Nutritional Value:")
                        nutrients = fallback_meal['nutrition_total']
                        print(f"  - Carbohydrates: {nutrients['carbohydrate']} g")
                        print(f"  - Proteins: {nutrients['protein']} g")
                        print(f"  - Fats: {nutrients['total_fat']} g")
                        print(f"  - Iron: {nutrients['iron_mg']} mg")
                        print()
                    else:
                        print(f"Warning: No unique {meal_type} meal available for user {user_id}, even with fallback.")
                else:
                    print(f"Warning: Fallback meal for {meal_type} contains restricted foods for user {user_id}.")
            elif not meal_generated:
                print(f"Warning: Could not generate unique {meal_type} meal for user {user_id} and no fallback available.")
    
    return recommendations

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
    gender = user_data['gender']
    all_recommendations = {}

    # Fetch and parse the latest report
    report = get_user_report(user_id)
    if report:
        print(f"Report: Cholesterol={report['cholesterol']} mg/dl, Sugar={report['sugar']} mg/dl, Hemoglobin={report['hemoglobin']} g/dl")
        health_status = get_health_status(report, gender)
        print(f"Health Status: {health_status}")
        daily_nutrients = adjust_nutrients(DAILY_NUTRIENTS, health_status)
    else:
        print("No valid report found, using default nutrients.")
        health_status = {}
        daily_nutrients = DAILY_NUTRIENTS
    
    print(f"Adjusted Daily Nutrients: {daily_nutrients}")

    # Train decision tree
    clf, feature_names = train_decision_tree(food_data, daily_nutrients)
    if clf is None:
        return jsonify({'error': 'Decision tree training failed'}), 500

    # Delete existing recommendations for this user per meal type
    for meal_type in MEAL_TYPES:
        delete_existing_recommendations(user_id, meal_type)

    # Generate and store new recommendations
    options = recommend_meals_for_user(user_id, food_data, clf, feature_names, restrictions, daily_nutrients)
    for meal_type in MEAL_TYPES:
        all_recommendations[meal_type] = [
            {
                'option': i + 1,
                'name': option['meal_name'],
                'ingredients': option['meal_ingredients'].split(', '),
                'nutritional_totals': json.loads(option['nutrition_total'])
            } for i, option in enumerate([rec for rec in options if rec['meal_type'] == meal_type])
        ]
        # Insert into Supabase
        for option in [rec for rec in options if rec['meal_type'] == meal_type]:
            try:
                supabase.table('personal_recommendation').insert(option).execute()
            except Exception as e:
                print(f"Error storing recommendation for {option['meal_name']}: {e}")

    response = {
        'user_id': user_id,
        'recommendations': all_recommendations,
        'execution_time': f"{time.time() - start_time:.2f} seconds"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
