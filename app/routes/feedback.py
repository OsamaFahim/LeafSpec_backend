from flask import Blueprint, request, jsonify
from app import mongo
from bson import ObjectId

feedback_bp = Blueprint('feedback_bp', __name__)

@feedback_bp.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    print(data)
    mongo['LeafSpec'].feedback.insert_one(data)
    return jsonify({'message': 'Feedback received!'}), 201

@feedback_bp.route('/feedback', methods=['GET'])
def get_feedback():
    feedback = mongo['LeafSpec'].feedback.find()
    return jsonify({'feedback': [f for f in feedback]}), 200