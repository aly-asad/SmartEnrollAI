# ============================================================
# SmartEnroll AI — Flask API
# This file is the bridge between the website and ML model
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

# Create the Flask app
app = Flask(__name__)

# Allow requests from any website (needed for Lovable.ai frontend)
CORS(app)

# ============================================================
# Load all saved ML files when the server starts
# ============================================================
print("Loading ML model files...")

model         = joblib.load('best_model.pkl')
scaler        = joblib.load('scaler.pkl')
career_encoder  = joblib.load('career_encoder.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

print("All files loaded successfully!")

# ============================================================
# Route 1 — Home (just to check API is running)
# ============================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message' : 'SmartEnroll AI API is running!',
        'status'  : 'active',
        'version' : '1.0'
    })

# ============================================================
# Route 2 — Predict Career (main endpoint)
# Website will send student data here and get career back
# ============================================================
@app.route('/predict', methods=['POST'])
def predict_career():
    try:
        # Get the student data sent from the website
        student_data = request.get_json()

        # --- Build input DataFrame ---
        input_data = {
            'Gender'                          : student_data['gender'],
            'School_Type'                     : student_data['school_type'],
            'Socioeconomic_Status'            : student_data['socioeconomic_status'],
            'Mathematics_Score'               : float(student_data['math_score']),
            'Science_Score'                   : float(student_data['science_score']),
            'Language_Arts_Score'             : float(student_data['language_arts_score']),
            'Social_Studies_Score'            : float(student_data['social_studies_score']),
            'Mathematics_Improvement'         : float(student_data['math_improvement']),
            'Science_Improvement'             : float(student_data['science_improvement']),
            'Language_Arts_Improvement'       : float(student_data['language_arts_improvement']),
            'Social_Studies_Improvement'      : float(student_data['social_studies_improvement']),
            'Logical_Reasoning'               : float(student_data['logical_reasoning']),
            'Critical_Thinking'               : float(student_data['critical_thinking']),
            'Analytical_Ability'              : float(student_data['analytical_ability']),
            'Creativity'                      : float(student_data['creativity']),
            'Communication'                   : float(student_data['communication']),
            'Emotional_Intelligence'          : float(student_data['emotional_intelligence']),
            'Social_Skills'                   : float(student_data['social_skills']),
            'Leadership'                      : float(student_data['leadership']),
            'Arts_Participation'              : float(student_data['arts_participation']),
            'Arts_Involvement'                : float(student_data['arts_involvement']),
            'Science_Club_Participation'      : float(student_data['science_club_participation']),
            'Science_Club_Involvement'        : float(student_data['science_club_involvement']),
            'Debate_Participation'            : float(student_data['debate_participation']),
            'Debate_Involvement'              : float(student_data['debate_involvement']),
            'Community_Service_Participation' : float(student_data['community_service_participation']),
            'Community_Service_Involvement'   : float(student_data['community_service_involvement']),
            'Learning_Style'                  : student_data['learning_style']
        }

        input_df = pd.DataFrame([input_data])

        # --- Encode text columns ---
        for col in ['Gender', 'School_Type', 'Socioeconomic_Status', 'Learning_Style']:
            input_df[col] = feature_encoders[col].transform(input_df[col])

        # --- Add engineered features (same as training) ---
        input_df['Academic_Average'] = (
            input_df['Mathematics_Score'] + input_df['Science_Score'] +
            input_df['Language_Arts_Score'] + input_df['Social_Studies_Score']
        ) / 4

        input_df['Technical_Ability'] = (
            input_df['Logical_Reasoning'] + input_df['Critical_Thinking'] +
            input_df['Analytical_Ability']
        ) / 3

        input_df['Social_Ability'] = (
            input_df['Communication'] + input_df['Emotional_Intelligence'] +
            input_df['Social_Skills']
        ) / 3

        input_df['Creative_Score'] = (
            input_df['Creativity'] + input_df['Arts_Participation'] +
            input_df['Arts_Involvement']
        ) / 3

        # --- Reorder columns to match training ---
        input_df = input_df[feature_columns]

        # --- Scale the input ---
        input_scaled = scaler.transform(input_df)

        # --- Get prediction ---
        prediction_encoded  = model.predict(input_scaled)[0]
        probabilities       = model.predict_proba(input_scaled)[0]
        predicted_career    = career_encoder.inverse_transform([prediction_encoded])[0]
        confidence          = round(float(probabilities[prediction_encoded]) * 100, 2)

        # --- Get top 3 career suggestions ---
        top3_indices = probabilities.argsort()[::-1][:3]
        top3_careers = [
            {
                'career'     : career_encoder.inverse_transform([i])[0],
                'probability': round(float(probabilities[i]) * 100, 2)
            }
            for i in top3_indices
        ]

        # --- Career descriptions for the website ---
        career_info = {
            'STEM'            : 'Computer Science, AI, Software Engineering, Mathematics',
            'Healthcare'      : 'Medicine, Nursing, Biology, Pharmacy',
            'Business_Finance': 'Business Administration, Finance, Marketing, Accounting',
            'Arts_Media'      : 'Design, Animation, Media, Fine Arts, Photography',
            'Education'       : 'Teaching, Academia, Training, Educational Psychology',
            'Government_Law'  : 'Law, Politics, Civil Services, Public Administration',
            'Social_Services' : 'Psychology, Social Work, Community Development, Counseling'
        }

        # --- Send result back to website ---
        return jsonify({
            'status'          : 'success',
            'predicted_career': predicted_career,
            'confidence'      : confidence,
            'description'     : career_info.get(predicted_career, ''),
            'top3_careers'    : top3_careers
        })

    except Exception as e:
        # If anything goes wrong, send error message
        return jsonify({
            'status' : 'error',
            'message': str(e)
        }), 400


# ============================================================
# Route 3 — Health Check (Render uses this to check if alive)
# ============================================================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


# ============================================================
# Run the app
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
