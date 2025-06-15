from app import mongo
from typing import List, Dict, Optional

class Species:
    def __init__(self, common_name=None, scientific_name=None, family=None):
        self.common_name = common_name
        self.scientific_name = scientific_name
        self.family = family
        self.db = mongo['LeafSpec']

    def to_dict(self) -> dict:
        """Convert species object to dictionary format for database storage"""
        return {
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "family": self.family,
            "origin": [],
            "type": None,
            "dimension": None,
            "cycle": None,
            "propagation": [],
            "hardiness": {
                "min": None,
                "max": None
            },
            "watering": None,
            "watering_general_benchmark": {
                "value": None,
                "unit": None
            },
            "plant_anatomy": [],
            "sunlight": [],
            "pruning_month": [],
            "soil": [],
            "drought_tolerant": False,
            "salt_tolerant": False,
            "indoor": False,
            "pest_susceptibility": [],
            "flowers": False,
            "flower_color": None,
            "fruits": False,
            "fruit_color": [],
            "harvest_season": None,
            "leaf": False,
            "leaf_color": [],
            "medicinal": False,
            "description": None,
            "default_image": {
                "original_url": None,
                "regular_url": None,
                "medium_url": None,
                "small_url": None,
                "thumbnail": None
            }
        }

    def save(self) -> str:
        """Save species to database"""
        if not self.common_name or not self.scientific_name or not self.family:
            raise ValueError("Common name, scientific name, and family are required")
        
        species_data = self.to_dict()
        species_id = self.db.species.insert_one(species_data).inserted_id
        return str(species_id)

    @staticmethod
    def find_by_common_name(common_name: str) -> Optional[Dict]:
        """Find a species by common name"""
        return mongo['LeafSpec'].species.find_one({"common_name": common_name})

    @staticmethod
    def find_by_scientific_name(scientific_name: str) -> Optional[Dict]:
        """Find a species by scientific name"""
        return mongo['LeafSpec'].species.find_one({"scientific_name": scientific_name})

    @staticmethod
    def exists(common_name: str) -> bool:
        """Check if a species exists by common name"""
        species = mongo['LeafSpec'].species.find_one({"common_name": common_name})
        return species is not None

    @classmethod
    def from_dict(cls, data: Dict) -> 'Species':
        """Create a Species instance from a dictionary"""
        species = cls(
            common_name=data.get('common_name'),
            scientific_name=data.get('scientific_name'),
            family=data.get('family')
        )
        return species

    def update(self, species_id: str, update_data: Dict) -> bool:
        """Update species information"""
        result = self.db.species.update_one(
            {"_id": species_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    @staticmethod
    def delete(species_id: str) -> bool:
        """Delete a species from database"""
        result = mongo['LeafSpec'].species.delete_one({"_id": species_id})
        return result.deleted_count > 0

    @staticmethod
    def get_all() -> List[Dict]:
        """Get all species with pagination"""
        cursor = mongo['LeafSpec'].species.find()
        return list(cursor)

    def add_image(self, species_id: str, image_data: Dict) -> bool:
        """Add or update default image for a species"""
        result = self.db.species.update_one(
            {"_id": species_id},
            {"$set": {"default_image": image_data}}
        )
        return result.modified_count > 0

    @staticmethod
    def search(query: str) -> List[Dict]:
        """Search species by common name or scientific name"""
        return list(mongo['LeafSpec'].species.find({
            "$or": [
                {"common_name": {"$regex": query, "$options": "i"}},
                {"scientific_name": {"$regex": query, "$options": "i"}}
            ]
        }))

    def validate(self) -> bool:
        """Validate species data"""
        if not self.common_name or not self.scientific_name or not self.family:
            return False
        return True