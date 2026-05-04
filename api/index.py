import json
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer
from http.server import BaseHTTPRequestHandler

# Load the trained model once at cold start
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Prepare descriptor metadata (same as training)
DESC_LIST = Descriptors.descList  # List of (name, function) tuples
DESC_NAMES = [d[0] for d in DESC_LIST]

def compute_descriptors(smiles):
    """Convert a SMILES string to a list of descriptor values."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [func(mol) for _, func in DESC_LIST]

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests from the browser"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_GET(self): 
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write("API is running. Please use POST for predictions.".encode())
        
    def do_POST(self):
        """Handle POST requests for predictions"""
        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))

            smiles = body.get('smiles')
            if not smiles:
                raise ValueError('Missing "smiles" in request payload.')

            # Compute descriptors
            desc_vals = compute_descriptors(smiles)
            if desc_vals is None:
                raise ValueError('Invalid SMILES string; could not be parsed by RDKit.')

            # Build DataFrame (single row)
            X = pd.DataFrame([desc_vals], columns=DESC_NAMES)

            # Replace infinities with NaN
            X = X.replace([np.inf, -np.inf], np.nan)

            # Impute missing values (mean strategy)
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            # Clip to float32 limits to avoid overflow in tree-based models
            X_clipped = np.clip(
                X_imputed,
                np.finfo(np.float32).min,
                np.finfo(np.float32).max
            )

            # Predict probability of the positive class (toxic)
            prob = model.predict_proba(X_clipped)[0][1]
            toxicity = 'Toxic' if prob >= 0.5 else 'Non-Toxic'

            response_body = {
                'toxicity': toxicity,
                'probability': float(prob)
            }

            # Return success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_body).encode('utf-8'))

        except Exception as e:
            # Return a 400 Bad Request with error details
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
