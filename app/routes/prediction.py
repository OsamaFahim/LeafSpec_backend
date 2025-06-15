from flask import Blueprint, request, jsonify
from app.services.prediction_service import PredictionService
from PIL import Image
import io
import cv2
import numpy as np

prediction_bp = Blueprint('prediction', __name__)

def is_image_blurry(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 10

# Now using binary validation + species classification approach

@prediction_bp.route('/predict_species', methods=['POST'])
def predict_species():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Unsupported file format'}), 400
        
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        # Check if image is too blurry
        if is_image_blurry(image):
            return jsonify({'error': 'Image is too blurry'}), 400
        
        # Get validation result from our refactored prediction service
        result = PredictionService.validate_and_predict(image)
        print("result" , result)

        if result["is_plant"]:
            # Get user email from request if available for database storage
            user_email = None
            if 'user_email' in request.form:
                user_email = request.form['user_email']
            
            # Save prediction to database
            prediction_id = PredictionService.save_prediction(
                image=image,
                predicted_species=result["predicted_species"],
                confidence=str(result["species_confidence"]),
                user_email=user_email
            )
            
            return jsonify({
                'is_plant': True,
                'binary_confidence': round(result["binary_confidence"], 4),
                'predicted_species': result["predicted_species"],
                'species_confidence': round(result["species_confidence"], 4),
                'prediction_id': prediction_id
            })
        else:
            return jsonify({
                'is_plant': False,
                'binary_confidence': round(result["binary_confidence"], 4),
                'message': "The uploaded image is not a plant leaf."
            })
            
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500
