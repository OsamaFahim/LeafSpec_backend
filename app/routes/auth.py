import os
from flask import Blueprint, request, jsonify
from app.models.user import User
from bson.objectid import ObjectId
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from bson import ObjectId

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/add_user', methods=['POST'])
def add_user():
    try:
        data = request.get_json()
        
        if User.exists(data.get('email')):
            return jsonify({"error": "User Already exists"}), 409
        
        user = User(
            name=data.get('name'),
            email=data.get('email'),
            password=data.get('password')
        )
        user_id = user.save()
        if not user_id:
            return jsonify({"error": "An error occurred"}), 500
        userJson = user.toJson()
        return jsonify({"user": userJson}), 200
    
    except ValueError as e:
        print(e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "An error occurred"}), 500

@auth_bp.route('/sign_in', methods=['POST'])
def sign_in():
    try:
        data = request.get_json()
        
        if not data.get('email') or not data.get('password'):
            return jsonify({"error": "Email and password are required"}), 400
        
        user = User.authenticate(data['email'], data['password'])
        
        if user:

            userJson = user
            userJson['_id'] = str(user['_id'])
            return jsonify({"user": userJson}), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401
            
    except Exception as e:
        print(e)
        return jsonify({"error": "An error occurred"}), 500
    
    
@auth_bp.route('/google_sign_in', methods=['POST'])
def google_sign_in():
    try:
        # Get the ID token from the request
        id_token_str = request.get_json().get('id_token')
        if not id_token_str:
            return jsonify({"error": "ID token is required"}), 400
        # Verify the Google ID token
        try:
            # Specify the CLIENT_ID from your Google Developer Console
            id_info = id_token.verify_oauth2_token(
                id_token_str, 
                google_requests.Request(), 
                os.getenv('GOOGLE_CLIENT_ID')
            )
        except ValueError:
            return jsonify({"error": "Invalid ID token"}), 401
        # Extract user information from the token
        email = id_info.get('email')
        name = id_info.get('name')
        if not email:
            return jsonify({"error": "Unable to retrieve email from Google"}), 400
        user = User.find_by_email(email)
        if (user):
            userJson = user
            userJson['_id'] = str(user['_id'])
            return jsonify({"user": userJson}), 200
        # Create a new user if they don't exist
        new_user = User(
            name=name,
            email=email,
            password=User.generate_random_password(),
            auth_type='google'
        )
        user_id = new_user.save()
        userJson = new_user.toJson()
        return jsonify({"user": userJson}), 200
    
    except Exception as e:
        # Log the error for debugging
        print(f"Google Sign-In Error: {str(e)}")
        return jsonify({"error": "An error occurred during Google Sign-In"}), 500
    

@auth_bp.route('/add_favourite', methods=['POST'])
def add_favourite():
    try:
        data = request.get_json()
        
        if not data.get('userEmail') or not data.get('favorites'):
            return jsonify({"error": "Email and species ID are required"}), 400
        print(data['userEmail'], data['favorites'])
        user = User.find_by_email(data['userEmail'])
        if not user:
            return jsonify({"error": "User not found"}), 404
        print(user)
        favourites = user['favourites'] if user.get('favourites') else []	
        print(favourites)
        new_fav = data['favorites']
        new_fav = new_fav.replace("[","").replace("]","").replace('"',"")
        print(new_fav,"qqqqq",type(new_fav))
        # for fav in new_fav.split(","):
        #     if fav not in favourites:
        #         favourites.append(fav)
        favourites = new_fav.split(",")
        print(favourites)
        User.update_favourites(data['userEmail'], favourites)
        print(favourites)
        return jsonify({"favourites": favourites}), 200
    
    except Exception as e:
        print(e)
        return jsonify({"error": "An error occurred"}), 500

