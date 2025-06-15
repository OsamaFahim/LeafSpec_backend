from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from app import mongo
from bson.objectid import ObjectId

class AnalyticsService:
    @staticmethod
    def get_prediction_count():
        """Get total number of predictions"""
        return mongo['LeafSpec'].predictions.count_documents({})
    
    @staticmethod
    def get_prediction_trends(days=30):
        """Get daily prediction counts for the specified number of days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Aggregate daily counts
        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff_date}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}  # Sort by date ascending
        ]
        
        results = list(mongo['LeafSpec'].predictions.aggregate(pipeline))
        
        # Format results
        dates = [(datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, 0, -1)]
        counts = [0] * len(dates)
        
        # Fill in actual counts where available
        date_to_index = {date: idx for idx, date in enumerate(dates)}
        for result in results:
            if result["_id"] in date_to_index:
                counts[date_to_index[result["_id"]]] = result["count"]
        
        return {"dates": dates, "counts": counts}
    
    @staticmethod
    def get_popular_species(limit=10):
        """Get most frequently predicted species"""
        pipeline = [
            {
                "$group": {
                    "_id": "$species",
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            },
            {
                "$limit": limit
            }
        ]
        
        results = list(mongo['LeafSpec'].predictions.aggregate(pipeline))
        return [{"species": item["_id"], "count": item["count"]} for item in results]
    
    @staticmethod
    def get_confidence_metrics():
        """Get confidence metrics for each species"""
        pipeline = [
            {
                "$group": {
                    "_id": "$species",
                    "avg_confidence": {"$avg": {"$toDouble": "$confidence"}},
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            }
        ]
        
        results = list(mongo['LeafSpec'].predictions.aggregate(pipeline))
        return [{"species": item["_id"], "avg_confidence": item["avg_confidence"], "count": item["count"]} for item in results]
    
    @staticmethod
    def get_prediction_history(page=1, per_page=20, filters=None):
        """Get paginated prediction history with optional filters"""
        skip = (page - 1) * per_page
        
        # Build query from filters
        query = {}
        if filters:
            if "species" in filters and filters["species"]:
                query["species"] = filters["species"]
            if "user_email" in filters and filters["user_email"]:
                query["user_email"] = filters["user_email"]
            if "date_from" in filters and filters["date_from"]:
                query.setdefault("timestamp", {})["$gte"] = datetime.fromisoformat(filters["date_from"])
            if "date_to" in filters and filters["date_to"]:
                query.setdefault("timestamp", {})["$lte"] = datetime.fromisoformat(filters["date_to"])
            if "min_confidence" in filters and filters["min_confidence"]:
                query["confidence"] = {"$gte": filters["min_confidence"]}
        
        # Get total count for pagination
        total = mongo['LeafSpec'].predictions.count_documents(query)
        
        # Get paginated results
        cursor = mongo['LeafSpec'].predictions.find(
            query,
            {"image_data": 0}  # Exclude image data for performance
        ).sort("timestamp", -1).skip(skip).limit(per_page)
        
        predictions = []
        for pred in cursor:
            pred["_id"] = str(pred["_id"])  # Convert ObjectId to string
            predictions.append(pred)
        
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page,
            "items": predictions
        }
    
    @staticmethod
    def get_prediction_by_id(prediction_id):
        """Get a prediction by ID"""
        try:
            pred = mongo['LeafSpec'].predictions.find_one({"_id": ObjectId(prediction_id)})
            if pred:
                pred["_id"] = str(pred["_id"])
            return pred
        except:
            return None
