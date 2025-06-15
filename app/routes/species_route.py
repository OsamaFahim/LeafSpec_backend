import os
from flask import Blueprint, request, jsonify
from app.models.species import Species
from bson.objectid import ObjectId
from bson import ObjectId

species_bp = Blueprint('species', __name__)

@species_bp.route('/get_specie', methods=['POST'])
def get_specie():
    try:
        data = request.get_json()
        
        if not data.get('specie'):
            return jsonify({"error": "name is required"}), 400
        
        specie = Species.find_by_common_name(data['specie'])
        if specie:
            return jsonify(specie), 200
        else:
            specie = Species.find_by_scientific_name(data['specie'])
            if specie:
                return jsonify(specie), 200
            else:
                return jsonify({"error": "Specie not found"}), 404
            
    except Exception as e:
        return jsonify({"error": "An error occurred"}), 500
    
@species_bp.route('/get_all_species', methods=['GET'])
def get_all_species():
    try:
        species = Species.get_all()
        print(species)
        for doc in species:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
        print(jsonify(species))
        return jsonify(species), 200
    except Exception as e:
        return jsonify({"error": "An error occurred"}), 500
