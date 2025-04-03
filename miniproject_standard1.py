from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.tree import DecisionTreeClassifier
import time
import re
import random
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Supabase credentials
url = "https://jujxoskixfadyvrxlaru.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1anhvc2tpeGZhZHl2cnhsYXJ1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjM4MjI5NSwiZXhwIjoyMDU3OTU4Mjk1fQ.Ka4RjOSpSr5ODKpklgvFJkx9iNPxgqwIMFLbQU5-NMo"
supabase: Client = create_client(url, key)

# Nutritional targets
targets = {
    'breakfast': {
        'carbohydrate': 90, 'protein': 30, 'total_fat': 24, 'dietary_fibre_total': 10,
        'total_ascorbic_acid': 20, 'calcium_mg': 300, 'iron_mg': 4 - 4, 'vite': 5,
        'linoleic_c18_2n6': 5, 'alpha_linolenic_c18_3n3': 0.5, 'lys': 1.5, 'met': 0.5
    },
    'lunch': {
        'carbohydrate': 120, 'protein': 40, 'total_fat': 32, 'dietary_fibre_total': 15,
        'total_ascorbic_acid': 30, 'calcium_mg': 400, 'iron_mg': 6, 'vite': 7,
        'linoleic_c18_2n6': 7, 'alpha_linolenic_c18_3n3': 0.7, 'lys': 2.0, 'met': 0.7
    },
    'dinner': {
        'carbohydrate': 90, 'protein': 30, 'total_fat': 24, 'dietary_fibre_total': 10,
        'total_ascorbic_acid': 20, 'calcium_mg': 300, 'iron_mg': 4, 'vite': 5,
        'linoleic_c18_2n6': 5, 'alpha_linolenic_c18_3n3': 0.5, 'lys': 1.5, 'met': 0.5
    }
}

# Fetch and preprocess data once at startup
print("Loading data...")
start_time = time.time()
tables = {
    "nutritionaldata": "nutritional_df",
    "watersolublevitamins": "vitamins_df",
    "amino_acid_profile": "amino_df",
    "carotenoids": "carotenoids_df",
    "edibleoils": "oils_df",
    "fat_soluble_vitamins": "fat_vitamins_df",
    "fatty_acid_profile": "fatty_acids_df",
    "minerals_trace_elements": "minerals_df",
    "oligosaccharides_phytosterols": "oligo_df",
    "organic_acids": "organic_df",
    "polyphenols": "polyphenols_df",
    "starch_and_sugars": "starch_df"
}

dataframes = {}
for table, df_name in tables.items():
    response = supabase.table(table).select("*").execute()
    dataframes[df_name] = pd.DataFrame(response.data)

combined_df = dataframes["nutritional_df"]
for df_name, df in dataframes.items():
    if df_name != "nutritional_df":
        combined_df = combined_df.merge(df, on='food_code', how='left', suffixes=('', f'_{df_name}'))
combined_df.fillna(0, inplace=True)
print(f"Data loaded in {time.time() - start_time:.2f} seconds")

# Helper functions
def get_user_restrictions(user_id):
    response = supabase.table("UserTable").select("notes").eq("auth_uid", user_id).execute()
    if response.data:
        notes = response.data[0].get('notes') or {}
        restrictions = notes.get('restrictions', []) if isinstance(notes, dict) else []
        return [r.lower() for r in restrictions if isinstance(r, str)]
    return []

def train_decision_tree(df, meal_type):
    features = ['carbohydrate', 'protein', 'total_fat', 'dietary_fibre_total',
               'total_ascorbic_acid', 'calcium_mg', 'iron_mg', 'vite',
               'linoleic_c18_2n6', 'alpha_linolenic_c18_3n3', 'lys', 'met']
    X = df[features].values
    y = np.array([1 if (row['carbohydrate'] > 0 and row['carbohydrate'] <= targets[meal_type]['carbohydrate'] * 1.5 and
                        row['protein'] > 0 and row['protein'] <= targets[meal_type]['protein'] * 1.5 and
                        row['total_fat'] > 0 and row['total_fat'] <= targets[meal_type]['total_fat'] * 1.5)
                  else 0
                  for _, row in df.iterrows()])
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    return clf

def meal_name_exists(user_id, meal_type, meal_name):
    try:
        response = supabase.table("standard_recommendation").select("meal_name")\
            .eq("user_id", user_id)\
            .eq("meal_type", meal_type)\
            .eq("meal_name", meal_name)\
            .execute()
        return len(response.data) > 0
    except Exception:
        return False

def generate_meal_name(user_id, meal_type, ingredients):
    if not ingredients:
        return f"{meal_type.capitalize()} Placeholder"
    def simplify_name(name):
        return re.split(r'[,(]', name)[0].strip()
    main_ingredients = [simplify_name(ingredients[i]) for i in range(min(2, len(ingredients)))]
    base_name = f"{meal_type.capitalize()} {main_ingredients[0]} & {main_ingredients[1]}"
    if not meal_name_exists(user_id, meal_type, base_name):
        return base_name
    descriptors = ['Delight', 'Rise & Shine', 'Feast', 'Glow', 'Harmony']
    for descriptor in descriptors:
        new_name = f"{base_name} {descriptor}"
        if not meal_name_exists(user_id, meal_type, new_name):
            return new_name
    if len(ingredients) >= 3:
        third_ingredient = simplify_name(ingredients[2])
        new_name = f"{base_name} with {third_ingredient}"
        if not meal_name_exists(user_id, meal_type, new_name):
            return new_name
    return f"{base_name} {random.randint(1, 100)}"

def recommend_meals_dt(user_id, meal_type, df, restrictions, num_meals=3):
    if restrictions:
        pattern = '|'.join([fr'\b{restr}\b' for restr in restrictions] + [fr'{restr}.*' for restr in restrictions])
        filtered_df = df[~df['food_name_nutri'].str.lower().str.contains(pattern, na=False, case=False)]
    else:
        filtered_df = df.copy()
    
    clf = train_decision_tree(filtered_df, meal_type)
    features = ['carbohydrate', 'protein', 'total_fat', 'dietary_fibre_total',
               'total_ascorbic_acid', 'calcium_mg', 'iron_mg', 'vite',
               'linoleic_c18_2n6', 'alpha_linolenic_c18_3n3', 'lys', 'met']
    X = filtered_df[features].values
    predictions = clf.predict(X)
    suitable_foods = filtered_df[predictions == 1].sample(frac=1, random_state=42).reset_index(drop=True)
    
    meal_options = []
    used_meal_sets = set()
    min_targets = {k: v * 0.85 for k, v in targets[meal_type].items()}
    max_targets = {k: v * 1.2 for k, v in targets[meal_type].items()}
    
    for meal_num in range(num_meals):
        meal = []
        totals = {key: 0 for key in targets[meal_type].keys()}
        shuffled_foods = suitable_foods.sample(frac=1, random_state=meal_num * 100).reset_index(drop=True)
        idx = 0
        while idx < len(shuffled_foods):
            row = shuffled_foods.iloc[idx]
            food = row.get('food_name_nutri') or row.get('food_name')
            temp_totals = totals.copy()
            for nutrient in totals.keys():
                temp_totals[nutrient] += row.get(nutrient, 0)
            if all(temp_totals[n] <= max_targets[n] for n in temp_totals):
                meal.append(food)
                totals = temp_totals
            idx += 1
            if all(totals[n] >= min_targets[n] for n in totals):
                meal_tuple = tuple(sorted(meal))
                if meal_tuple not in used_meal_sets:
                    used_meal_sets.add(meal_tuple)
                    meal_name = generate_meal_name(user_id, meal_type, meal)
                    meal_options.append({'meal': meal, 'totals': totals, 'name': meal_name})
                    break
        if len(meal_options) <= meal_num and idx >= len(shuffled_foods):
            shuffled_foods = suitable_foods.sample(frac=1, random_state=meal_num * 200).reset_index(drop=True)
            idx = 0
            while idx < len(shuffled_foods):
                row = shuffled_foods.iloc[idx]
                food = row.get('food_name_nutri') or row.get('food_name')
                temp_totals = totals.copy()
                for nutrient in totals.keys():
                    temp_totals[nutrient] += row.get(nutrient, 0)
                if all(temp_totals[n] <= max_targets[n] for n in temp_totals):
                    meal.append(food)
                    totals = temp_totals
                idx += 1
                if all(totals[n] >= min_targets[n] for n in totals):
                    meal_tuple = tuple(sorted(meal))
                    if meal_tuple not in used_meal_sets:
                        used_meal_sets.add(meal_tuple)
                        meal_name = generate_meal_name(user_id, meal_type, meal)
                        meal_options.append({'meal': meal, 'totals': totals, 'name': meal_name})
                        break
        if len(meal_options) <= meal_num:
            meal_tuple = tuple(sorted(meal))
            if meal_tuple not in used_meal_sets:
                used_meal_sets.add(meal_tuple)
                meal_name = generate_meal_name(user_id, meal_type, meal)
                meal_options.append({'meal': meal, 'totals': totals, 'name': meal_name})
    
    while len(meal_options) < num_meals:
        base_meal = meal_options[0]['meal'].copy()
        base_totals = meal_options[0]['totals'].copy()
        shuffled_foods = suitable_foods.sample(frac=1, random_state=len(meal_options) * 300).reset_index(drop=True)
        extra_food = shuffled_foods.iloc[0].get('food_name_nutri') or shuffled_foods.iloc[0].get('food_name')
        new_meal = base_meal + [extra_food]
        for nutrient in base_totals.keys():
            base_totals[nutrient] += shuffled_foods.iloc[0].get(nutrient, 0)
        meal_tuple = tuple(sorted(new_meal))
        if meal_tuple not in used_meal_sets:
            used_meal_sets.add(meal_tuple)
            meal_name = generate_meal_name(user_id, meal_type, new_meal)
            meal_options.append({'meal': new_meal, 'totals': base_totals, 'name': meal_name})
    
    return meal_options[:num_meals]

def delete_and_insert_recommendations(user_id, meal_type, options):
    for option in options:
        data = {
            'user_id': user_id,
            'meal_type': meal_type,
            'meal_name': option['name'],
            'meal_ingredients': json.dumps(option['meal']),
            'nutrition_total': json.dumps({k: round(float(v), 2) for k, v in option['totals'].items()})
        }
        try:
            supabase.table("standard_recommendation")\
                .delete()\
                .eq("user_id", user_id)\
                .eq("meal_type", meal_type)\
                .eq("meal_name", option['name'])\
                .execute()
            supabase.table("standard_recommendation")\
                .insert(data)\
                .execute()
        except Exception as e:
            print(f"Error storing recommendation: {str(e)}")

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

    for meal_type in ['breakfast', 'lunch', 'dinner']:
        options = recommend_meals_dt(user_id, meal_type, combined_df, restrictions, num_meals)
        delete_and_insert_recommendations(user_id, meal_type, options)
        all_recommendations[meal_type] = [
            {
                'option': i + 1,
                'name': option['name'],
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
    app.run(host='0.0.0.0', port=5000, debug=True)