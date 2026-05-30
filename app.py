from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

print("Loading ML model files...")
model            = joblib.load('best_model.pkl')
scaler           = joblib.load('scaler.pkl')
career_encoder   = joblib.load('career_encoder.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')
feature_columns  = joblib.load('feature_columns.pkl')
print("All files loaded successfully!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'SmartEnroll AI API is running!', 'status': 'active', 'version': '1.0'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict_career():
    try:
        d = request.get_json()

        input_data = {
            'Gender'                          : d['gender'],
            'School_Type'                     : d['school_type'],
            'Socioeconomic_Status'            : d['socioeconomic_status'],
            'Mathematics_Score'               : float(d['math_score']),
            'Science_Score'                   : float(d['science_score']),
            'Language_Arts_Score'             : float(d['language_arts_score']),
            'Social_Studies_Score'            : float(d['social_studies_score']),
            'Mathematics_Improvement'         : float(d['math_improvement']),
            'Science_Improvement'             : float(d['science_improvement']),
            'Language_Arts_Improvement'       : float(d['language_arts_improvement']),
            'Social_Studies_Improvement'      : float(d['social_studies_improvement']),
            'Logical_Reasoning'               : float(d['logical_reasoning']),
            'Critical_Thinking'               : float(d['critical_thinking']),
            'Analytical_Ability'              : float(d['analytical_ability']),
            'Creativity'                      : float(d['creativity']),
            'Communication'                   : float(d['communication']),
            'Emotional_Intelligence'          : float(d['emotional_intelligence']),
            'Social_Skills'                   : float(d['social_skills']),
            'Leadership'                      : float(d['leadership']),
            'Arts_Participation'              : float(d['arts_participation']),
            'Arts_Involvement'                : float(d['arts_involvement']),
            'Science_Club_Participation'      : float(d['science_club_participation']),
            'Science_Club_Involvement'        : float(d['science_club_involvement']),
            'Debate_Participation'            : float(d['debate_participation']),
            'Debate_Involvement'              : float(d['debate_involvement']),
            'Community_Service_Participation' : float(d['community_service_participation']),
            'Community_Service_Involvement'   : float(d['community_service_involvement']),
            'Learning_Style'                  : d['learning_style'],
            'STEM_Score'                      : float(d['stem_score']),
            'Business_Finance_Score'          : float(d['business_finance_score']),
            'Arts_Media_Score'                : float(d['arts_media_score']),
            'Healthcare_Score'                : float(d['healthcare_score']),
            'Education_Score'                 : float(d['education_score']),
            'Social_Services_Score'           : float(d['social_services_score']),
            'Government_Law_Score'            : float(d['government_law_score']),
        }

        input_df = pd.DataFrame([input_data])

        for col in ['Gender', 'School_Type', 'Socioeconomic_Status', 'Learning_Style']:
            input_df[col] = feature_encoders[col].transform(input_df[col])

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

        input_df    = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)

        prediction_encoded = model.predict(input_scaled)[0]
        probabilities      = model.predict_proba(input_scaled)[0]
        predicted_career   = career_encoder.inverse_transform([prediction_encoded])[0]
        confidence         = round(float(probabilities[prediction_encoded]) * 100, 2)

        top3_indices = probabilities.argsort()[::-1][:3]
        top3_careers = [
            {
                'career'      : career_encoder.inverse_transform([i])[0],
                'probability' : round(float(probabilities[i]) * 100, 2)
            }
            for i in top3_indices
        ]

        career_info = {
            'STEM'            : 'Computer Science, AI, Software Engineering, Mathematics',
            'Healthcare'      : 'Medicine, Nursing, Biology, Pharmacy',
            'Business_Finance': 'Business Administration, Finance, Marketing, Accounting',
            'Arts_Media'      : 'Design, Animation, Media, Fine Arts, Photography',
            'Education'       : 'Teaching, Academia, Training, Educational Psychology',
            'Government_Law'  : 'Law, Politics, Civil Services, Public Administration',
            'Social_Services' : 'Psychology, Social Work, Community Development, Counseling'
        }

        return jsonify({
            'status'          : 'success',
            'predicted_career': predicted_career,
            'confidence'      : confidence,
            'description'     : career_info.get(predicted_career, ''),
            'top3_careers'    : top3_careers
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
