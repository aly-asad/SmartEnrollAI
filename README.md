# SmartEnroll AI — Flask API

## Files Needed in This Folder
- app.py
- requirements.txt
- render.yaml
- best_model.pkl
- scaler.pkl
- career_encoder.pkl
- feature_encoders.pkl
- feature_columns.pkl

## How to Deploy on Render.com
1. Upload all files to GitHub
2. Go to render.com
3. New Web Service → Connect GitHub repo
4. It will auto deploy!

## API Endpoints
- GET  /         → Check if API is running
- POST /predict  → Send student data, get career prediction
- GET  /health   → Health check
