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

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Supabase configuration
SUPABASE_URL = "https://jujxoskixfadyvrxlaru.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1anhvc2tpeGZhZHl2cnhsYXJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIzODIyOTUsImV4cCI6MjA1Nzk1ODI5NX0.WcDFgHAMGbsksGUto44U4ke33yz_hONETzQX3U6-VcQ"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Nutritional requirements (daily values, simplified)
DAILY_NUTRIENTS = {
    'carbohydrate': 300,  # grams
    'protein': 50,        # grams
    'total_fat': 70,      # grams
    'iron_mg': 18,        # mg
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

# Convert preloaded data to dictionary format, including all food_codes from both tables
food_data = {}

# Step 1: Process nutritionaldata
nutri_df = dataframes['nutritionaldata_df']
for _, row in nutri_df.iterrows():
    food_code = row['food_code']
    food_data[food_code] = {
        'food_name': row['food_name'] or f"Unnamed_{food_code}",
        'carbohydrate': row.get('carbohydrate', None),
        'protein': row.get('protein', None),
        'total_fat': row.get('total_fat', None),
        'iron_mg': None  # Initially set to None, will be updated if present in minerals_trace_elements
    }

# Step 2: Process minerals_trace_elements and merge or add new entries
minerals_df = dataframes['minerals_trace_elements_df']
for _, row in minerals_df.iterrows():
    food_code = row['food_code']
    if food_code in food_data:
        # Update existing entry with mineral data
        food_data[food_code].update({
            'iron_mg': row.get('iron_mg', None)
        })
    else:
        # Add new entry for food_code unique to minerals_trace_elements
        food_data[food_code] = {
            'food_name': row['food_name'] or f"Unnamed_{food_code}",
            'carbohydrate': None,
            'protein': None,
            'total_fat': None,
            'iron_mg': row.get('iron_mg', None)
        }

print(f"Data loaded in {time.time() - start_time:.2f} seconds. Loaded {len(food_data)} food items.")

# Fetch restrictions for a specific user
def get_user_restrictions(user_id):
    response = supabase.table('UserTable').select('notes').eq('auth_uid', user_id).execute()
    if response.data:
        notes = response.data[0].get('notes') or {}
        restrictions = notes if isinstance(notes, list) else []
        return [r.lower() for r in restrictions if isinstance(r, str)]
    return []

# Delete existing recommendations for a user and meal type
def delete_existing_recommendations(user_id, meal_type):
    try:
        response = supabase.table('standard_recommendation').delete().eq('user_id', user_id).eq('meal_type', meal_type).execute()
        print(f"Deleted existing recommendations for user {user_id} and meal type {meal_type}.")
    except Exception as e:
        print(f"Error deleting existing recommendations: {e}")

# Decision tree to classify if a meal meets nutritional balance
def train_decision_tree(food_data):
    X = []
    y = []
    feature_names = ['carbohydrate', 'protein', 'total_fat', 'iron_mg']
    
    for food_code, nutrients in food_data.items():
        features = [nutrients.get(feat, 0) if nutrients.get(feat) is not None else 0 for feat in feature_names]
        X.append(features)
        balanced_count = sum(1 for feat, val in zip(feature_names, features) if val > 0 and DAILY_NUTRIENTS[feat] * 0.1 <= val <= DAILY_NUTRIENTS[feat] * 0.5)
        y.append(1 if balanced_count >= 2 else 0)
    
    if not X:
        print("ERROR: No training data for decision tree.")
        return None, feature_names
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    return clf, feature_names

# Generate a unique meal, respecting user restrictions
def generate_meal(food_data, clf, feature_names, used_ingredients, restricted_foods):
    available_foods = {
        k: v for k, v in food_data.items() 
        if k not in used_ingredients and v['food_name'].lower() not in restricted_foods
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

# Recommend meals for a user, respecting their restrictions
def recommend_meals_for_user(user_id, food_data, clf, feature_names, restricted_foods):
    if user_id not in user_meal_combinations:
        user_meal_combinations[user_id] = set()
    
    used_combinations = user_meal_combinations[user_id]
    recommendations = []
    
    for meal_type in MEAL_TYPES:
        print(f"\n{meal_type.capitalize()} Meals:")
        attempts = 0
        max_attempts = 10
        
        for i in range(MEALS_PER_TYPE):
            meal_generated = False
            while attempts < max_attempts:
                meal = generate_meal(food_data, clf, feature_names, set(), restricted_foods)
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
                if not any(ingredient.lower() in restricted_foods for ingredient in fallback_meal['meal_ingredients']):
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

# API Endpoint to trigger recommendations
@app.route('/api/generate-recommendations', methods=['POST'])
def generate_recommendations():
    data = request.get_json()
    user_id = data.get('user_id')
    num_meals = data.get('num_meals', 3)

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    start_time = time.time()
    restrictions = get_user_restrictions(user_id)
    all_recommendations = {}

    # Train decision tree once per request
    clf, feature_names = train_decision_tree(food_data)
    if clf is None:
        return jsonify({'error': 'Decision tree training failed'}), 500

    # Delete existing recommendations for this user
    user_ids = [user_id]
    if user_ids:
        for meal_type in MEAL_TYPES:
            delete_existing_recommendations(user_id, meal_type)

    # Generate and store new recommendations
    options = recommend_meals_for_user(user_id, food_data, clf, feature_names, restrictions)
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
                supabase.table('standard_recommendation').insert(option).execute()
            except Exception as e:
                print(f"Error storing recommendation: {str(e)}")

    response = {
        'user_id': user_id,
        'recommendations': all_recommendations,
        'execution_time': f"{time.time() - start_time:.2f} seconds"
    }
    return jsonify(response)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for production
