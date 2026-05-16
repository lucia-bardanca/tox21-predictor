import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Fragments

app = Flask(__name__)

# Cargar el modelo en memoria al iniciar la función serverless
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tox21_model_sr_mmp.joblib')
saved_data = joblib.load(MODEL_PATH)

def extract_all_features(smiles):
    # La misma función que usaste para entrenar
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp_array = generator.GetFingerprintAsNumPy(mol).astype(float) 
    
    frag_counts = [float(func(mol)) if name.startswith('fr_') else np.nan for name, func in Fragments.__dict__.items() if name.startswith('fr_')]
    physchem_desc = [float(func(mol)) if func(mol) is not None and not np.isnan(func(mol)) and not np.isinf(func(mol)) else np.nan for name, func in Descriptors.descList]
            
    return np.concatenate([fp_array, np.array(frag_counts), np.array(physchem_desc)])

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    smiles = data.get('smiles', '')
    
    features = extract_all_features(smiles)
    if features is None:
        return jsonify({'error': 'SMILES inválido o no se pudo procesar la molécula.'}), 400
        
    X = np.array(features).reshape(1, -1)
    
    try:
        # Aplicar el mismo pipeline de preprocesamiento
        X_prep = saved_data['imputer'].transform(X)
        X_prep = saved_data['v_selector'].transform(X_prep)
        X_prep = np.delete(X_prep, saved_data['corr_to_drop'], axis=1)
        X_scaled = saved_data['scaler'].transform(X_prep)
        X_final = X_scaled[:, saved_data['boruta'].support_]
        
        # Predicción
        prob = float(saved_data['model'].predict_proba(X_final)[:, 1][0])
        threshold = saved_data['threshold']
        is_toxic = bool(prob >= threshold)
        
        return jsonify({
            'smiles': smiles,
            'is_toxic': is_toxic,
            'probability': prob,
            'threshold_used': threshold
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Necesario para Vercel Serverless
def handler(request, response):
    return app(request, response)