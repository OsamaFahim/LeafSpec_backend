from PIL import Image
import numpy as np
from app import mb

def process_model_results(results):
    """
    Helper function to process ModelBit API results
    """
    if "error" in results:
        print(f"Error during prediction: {results['error']}")
        return None
    
    if "predictions" in results:
        predictions = results["predictions"]
        
        print("Top predictions:")
        for i, pred in enumerate(predictions):
            print(f"  {i+1}. {pred['species']} - {pred['confidence']:.2f}%")
        
        top_species = predictions[0]['species']
        top_confidence = predictions[0]['confidence']
        
        return {
            "top_species": top_species,
            "top_confidence": top_confidence,
            "all_predictions": predictions
        }

    return None

class PredictionService:
    """
    Service class for plant species prediction using either ModelBit API or local model
    """
    
    SPECIES = [
        "Ailanthus altissima Mill Swingle(182)",
        "Aloe Vera",
        "Alstonia Scholaris",
        "Apple",
        "Arjun",
        "Blueberry",
        "Buxus sempervirens L(200)",
        "Cherry",
        "Corn",
        "Corylus avellana L(199)",
        "Cotinus coggygria Scop(200)",
        "Crataegus monogyna Jacq(200)",
        "Fraxinus angustifolia Vahi(200)",
        "Grape",
        "Guava",
        "Hedera helix L(200)",
        "Jamun",
        "Jatropha",
        "Kale",
        "Laurus nobilis L(200)",
        "Lemon",
        "Mango",
        "Orange",
        "Peach",
        "Pepper Bell",
        "Phillyrea angustifolia L(200)",
        "Pistacia lentiscus L(200)",
        "Pittosporum tobira Thumb WTAiton(200)",
        "Pomegranate",
        "Pongamia Pinnata",
        "Populus alba L(200)",
        "Populus nigra L(200)",
        "Potato",
        "Quercus ilex L(200)",
        "Raspberry",
        "Ruscus aculeatus L(200)",
        "Soybean",
        "Strawberry",
        "Tomato"
    ]

    # Class variables to cache the models
    binary_classifier_model = None
    plant_classifier_model = None

    @staticmethod
    def preprocess_image_resnet(image):
        """
        Preprocess image for ResNet model using standard ImageNet preprocessing
        """
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)

    @staticmethod
    def predict_species_modelbit(image):
        """
        Predict plant species using ModelBit API with ResNet34 model
        """
        print("Predicting species using ModelBit API")
        try:
            input_tensor = PredictionService.preprocess_image_resnet(image)
            print("Image preprocessed for ResNet")
            
            tensor_array = input_tensor.cpu().numpy()
            
            response = mb.get_inference(
                region="us-east-1.aws",
                workspace="nu-edu",
                deployment="predict_species_using_resnet34",
                data={"image": tensor_array.tolist()},
                timeout=30
            )
            
            if isinstance(response, dict) and 'data' in response:
                if 'error' in response['data']:
                    error_msg = response['data']['error']
                    print(f"ModelBit API Error: {error_msg}")
                    return None, None
                
                result = process_model_results(response['data'])
                if result:
                    return result['top_species'], str(result['top_confidence'])
                
                if 'prediction' in response['data']:
                    prediction = response['data']['prediction']
                    if isinstance(prediction, list) and len(prediction) > 0:
                        predicted_index = np.argmax(prediction[0])
                        predicted_species = PredictionService.SPECIES[predicted_index]
                        confidence = prediction[0][predicted_index]
                        return predicted_species, str(confidence)
            
            print("Unexpected response format from ModelBit API")
            return None, None
            
        except Exception as e:
            print(f"Exception in ModelBit prediction: {str(e)}")
            return None, None

    @staticmethod
    def load_binary_classifier(model_path):
        """
        Load binary classifier model (ResNet18 with sigmoid output)
        """
        import torch
        import torch.nn as nn
        import torchvision.models as models
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def load_plant_classifier(model_path):
        """
        Load plant species classifier model (ResNet34)
        """
        import torch
        import torchvision.models as models
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(PredictionService.SPECIES))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def preprocess_image_local(image):
        """
        Preprocess image for local models (224x224 resize)
        """
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = image.convert("RGB")
        return transform(image).unsqueeze(0)

    @staticmethod
    def predict_species_only(image_tensor, model):
        """
        Helper function for species classification
        """
        import torch
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = PredictionService.SPECIES[predicted_idx.item()]
            return predicted_class, confidence.item()

    @staticmethod
    def _load_models():
        """
        Internal method to load both binary and plant classifier models
        """
        import os
        
        # Load binary classifier if not cached
        if PredictionService.binary_classifier_model is None:
            print("Loading binary classifier model")
            
            binary_model_paths = [
                "D:\\university\\fyp\\fyp app\\leafspec\\backend\\leaf_spec\\app\\static\\models\\leaf_resnet18_finetuned.pth",
                "../lib/models/Binary_classifier_ResNet18.pth",
                "app/static/models/Binary_classifier_ResNet18.pth"
            ]
            
            binary_model_path = None
            for path in binary_model_paths:
                if os.path.exists(path):
                    binary_model_path = path
                    break
            
            if not binary_model_path:
                raise Exception("Binary classifier model not found")
            
            PredictionService.binary_classifier_model = PredictionService.load_binary_classifier(binary_model_path)
            print("Binary classifier loaded and cached successfully")

        # Load plant classifier if not cached
        if PredictionService.plant_classifier_model is None:
            print("Loading plant classifier model")
            
            plant_model_paths = [
                "D:\\university\\fyp\\fyp app\\leafspec\\backend\\leaf_spec\\app\\static\\models\\resnet34_plant_classifier.pth",
                "../lib/models/resnet34_plant_classifier.pth",
                "app/static/models/resnet34_plant_classifier.pth",
                "D:\\university\\fyp\\fyp code files\\resnet34_plant_classifier3.pth",
                "D:\\university\\fyp\\fyp code files\\resnet34_plant_classifier.pth"
            ]
            
            plant_model_path = None
            for path in plant_model_paths:
                if os.path.exists(path):
                    plant_model_path = path
                    break
            
            if not plant_model_path:
                raise Exception("Plant classifier model not found")
            
            PredictionService.plant_classifier_model = PredictionService.load_plant_classifier(plant_model_path)
            print("Plant classifier loaded and cached successfully")

    @staticmethod
    def validate_and_predict(image):
        """
        Main validation and prediction method
        
        Returns:
            dict: Full validation and prediction results
        """
        import torch
        
        print("Validating and predicting species using local binary + plant classifiers")
        try:
            # Load both models
            PredictionService._load_models()

            # Preprocess the image
            image_tensor = PredictionService.preprocess_image_local(image)
            print("Image preprocessed for validation")

            # Step 1: Binary classification
            with torch.no_grad():
                binary_output = PredictionService.binary_classifier_model(image_tensor)
                binary_prob = binary_output.item()

            # Invert probability if needed (adjust based on your model training)
            binary_prob = 1 - binary_prob

            print(f"Binary classification result: {binary_prob:.4f} (threshold: 0.5)")

            # Step 2: Predict species if it's a plant
            if binary_prob >= 0.5:
                predicted_class, confidence = PredictionService.predict_species_only(
                    image_tensor, PredictionService.plant_classifier_model
                )
                print(f"Plant detected! Species: {predicted_class}, Confidence: {confidence:.4f}")
                
                return {
                    "is_plant": True,
                    "binary_confidence": binary_prob,
                    "predicted_species": predicted_class,
                    "species_confidence": confidence
                }
            else:
                print("Not a plant image detected")
                
                return {
                    "is_plant": False,
                    "binary_confidence": binary_prob,
                    "predicted_species": None,
                    "species_confidence": None
                }
                
        except Exception as e:
            import traceback
            print(f"Error in validation and prediction: {str(e)}")
            print(traceback.format_exc())
            raise e

    @staticmethod
    def predict_species_local(image):
        """
        Local prediction method for backward compatibility
        
        Returns:
            tuple: (predicted_species, confidence_score) or (None, None) if not a plant
        """
        try:
            result = PredictionService.validate_and_predict(image)
            
            if result["is_plant"]:
                return result["predicted_species"], str(result["species_confidence"])
            else:
                return None, None
                
        except Exception as e:
            print(f"Error in local prediction: {str(e)}")
            return None, None

    @staticmethod
    def save_prediction(image, predicted_species, confidence, user_email=None):
        """
        Save prediction results to the database
        """
        from app import mongo
        import datetime
        import base64
        from io import BytesIO
        
        print("Saving prediction to database")
        try:
            # Convert image to base64 string for storage
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Create prediction document
            prediction_data = {
                "species": predicted_species,
                "confidence": confidence,
                "timestamp": datetime.datetime.utcnow(),
                "image_data": img_str,
                "user_email": user_email
            }
            
            # Insert into predictions collection
            result = mongo['LeafSpec'].predictions.insert_one(prediction_data)
            print(f"Prediction saved with ID: {result.inserted_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return None

