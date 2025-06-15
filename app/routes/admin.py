from flask import Blueprint, request, jsonify, render_template, abort, flash, redirect, url_for
from app.services.analytics_service import AnalyticsService
from app.models.species import Species
from app.services.prediction_service import PredictionService
from bson.objectid import ObjectId
from app import mongo
import datetime
from functools import wraps
import json
import base64
from PIL import Image
import io
import os

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Admin authentication decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # # Check if user is admin - this is a placeholder for your actual auth logic
        # # You should replace this with your actual authentication system
        # admin_emails = ['admin@example.com','i212582@nu.edu.pk']  # List of admin emails
        # user_email = request.cookies.get('user_email')
        
        # if not user_email or user_email not in admin_emails:
        #     return abort(403)  # Forbidden
        return f(*args, **kwargs)
    return decorated_function

# Dashboard home
@admin_bp.route('/')
@admin_required
def dashboard():
    # Get summary statistics for the dashboard
    stats = {
        'total_predictions': AnalyticsService.get_prediction_count(),
        'total_users': mongo['LeafSpec'].users.count_documents({}),
        'total_species': mongo['LeafSpec'].species.count_documents({}),
        'total_feedback': mongo['LeafSpec'].feedback.count_documents({})
    }
    
    # Get recent predictions
    recent_predictions = AnalyticsService.get_prediction_history(page=1, per_page=5)['items']
    
    # Get popular species
    popular_species = AnalyticsService.get_popular_species(limit=5)
    
    return render_template('admin/dashboard.html', 
                           stats=stats,
                           recent_predictions=recent_predictions,
                           popular_species=popular_species)

# Prediction Analytics
@admin_bp.route('/predictions')
@admin_required
def predictions():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    # Get filters from query parameters
    filters = {}
    if request.args.get('species'):
        filters['species'] = request.args.get('species')
    if request.args.get('user_email'):
        filters['user_email'] = request.args.get('user_email')
    if request.args.get('date_from'):
        filters['date_from'] = request.args.get('date_from')
    if request.args.get('date_to'):
        filters['date_to'] = request.args.get('date_to')
    
    # Get prediction history
    history = AnalyticsService.get_prediction_history(
        page=page,
        per_page=per_page,
        filters=filters
    )
    
    # Get all species for filter dropdown
    all_species = [s['common_name'] for s in Species.get_all()]
    
    return render_template('admin/predictions.html', 
                          history=history,
                          all_species=all_species,
                          filters=filters)

# Get prediction detail
@admin_bp.route('/predictions/<prediction_id>')
@admin_required
def prediction_detail(prediction_id):
    prediction = AnalyticsService.get_prediction_by_id(prediction_id)
    if not prediction:
        abort(404)
    return render_template('admin/prediction_detail.html', prediction=prediction)

# Species Management
@admin_bp.route('/species')
@admin_required
def species_list():
    species_list = Species.get_all()
    return render_template('admin/species_list.html', species=species_list)

@admin_bp.route('/species/add', methods=['GET', 'POST'])
@admin_required
def add_species():
    if request.method == 'POST':
        common_name = request.form.get('common_name')
        scientific_name = request.form.get('scientific_name')
        family = request.form.get('family')
        description = request.form.get('description')
        
        # Create new species
        species = Species(common_name=common_name, scientific_name=scientific_name, family=family)
        species.description = description
        
        # Process uploaded image if provided
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                # Save image data to species
                image = Image.open(image_file)
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                species.image_data = img_str
        
        try:
            species_id = species.save()
            flash('Species added successfully!', 'success')
            return redirect(url_for('admin.species_list'))
        except Exception as e:
            flash(f'Error adding species: {str(e)}', 'error')
    
    return render_template('admin/species_form.html')

@admin_bp.route('/species/edit/<species_id>', methods=['GET', 'POST'])
@admin_required
def edit_species(species_id):
    species_data = mongo['LeafSpec'].species.find_one({"_id": ObjectId(species_id)})
    if not species_data:
        abort(404)
    
    if request.method == 'POST':
        update_data = {
            'common_name': request.form.get('common_name'),
            'scientific_name': request.form.get('scientific_name'),
            'family': request.form.get('family'),
            'description': request.form.get('description')
        }
        
        # Process uploaded image if provided
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                # Save image data
                image = Image.open(image_file)
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                update_data['image_data'] = img_str
        
        try:
            species = Species()
            result = species.update(species_id, update_data)
            if result:
                flash('Species updated successfully!', 'success')
            else:
                flash('No changes made to species.', 'info')
            return redirect(url_for('admin.species_list'))
        except Exception as e:
            flash(f'Error updating species: {str(e)}', 'error')
    
    return render_template('admin/species_form.html', species=species_data)

@admin_bp.route('/species/delete/<species_id>', methods=['POST'])
@admin_required
def delete_species(species_id):
    try:
        result = Species.delete(species_id)
        if result:
            flash('Species deleted successfully!', 'success')
        else:
            flash('Failed to delete species.', 'error')
    except Exception as e:
        flash(f'Error deleting species: {str(e)}', 'error')
    
    return redirect(url_for('admin.species_list'))

# Feedback Management
@admin_bp.route('/feedback')
@admin_required
def feedback_list():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    skip = (page - 1) * per_page
    
    # Get total count
    total = mongo['LeafSpec'].feedback.count_documents({})
    
    # Get paginated feedback
    feedback = list(mongo['LeafSpec'].feedback.find().sort("timestamp", -1).skip(skip).limit(per_page))
    for item in feedback:
        item['_id'] = str(item['_id'])
    
    pagination = {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }
    
    return render_template('admin/feedback.html', feedback=feedback, pagination=pagination)

@admin_bp.route('/feedback/resolve/<feedback_id>', methods=['POST'])
@admin_required
def mark_feedback_resolved(feedback_id):
    """Mark feedback as resolved"""
    try:
        result = mongo['LeafSpec'].feedback.update_one(
            {"_id": ObjectId(feedback_id)},
            {"$set": {"resolved": True, "resolved_at": datetime.datetime.utcnow()}}
        )
        
        if result.modified_count > 0:
            flash('Feedback marked as resolved', 'success')
        else:
            flash('Failed to update feedback', 'error')
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        
    return redirect(url_for('admin.feedback_list'))

# Performance Monitoring
@admin_bp.route('/performance')
@admin_required
def performance():
    # Get confidence metrics for all species
    confidence_metrics = AnalyticsService.get_confidence_metrics()
    
    # Get popular species
    popular_species = AnalyticsService.get_popular_species()
    
    return render_template('admin/performance.html', 
                          confidence_metrics=confidence_metrics,
                          popular_species=popular_species)

# System Health
@admin_bp.route('/system')
@admin_required
def system_health():
    # Get system metrics
    db_stats = mongo['LeafSpec'].command('dbStats')
    
    # Get API usage stats (last 7 days)
    today = datetime.datetime.now()
    week_ago = today - datetime.timedelta(days=7)
    
    api_stats = {
        'predictions': mongo['LeafSpec'].predictions.count_documents({"timestamp": {"$gte": week_ago}}),
        'user_registrations': mongo['LeafSpec'].users.count_documents({"registration_date": {"$gte": week_ago}}),
        'feedback': mongo['LeafSpec'].feedback.count_documents({"timestamp": {"$gte": week_ago}})
    }
    
    return render_template('admin/system.html', db_stats=db_stats, api_stats=api_stats)

@admin_bp.route('/api/predictions-chart-data')
@admin_required
def predictions_chart_data():
    """Get daily predictions data for charts"""
    days = 14  # Last 14 days
    
    # Generate dates
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [(today - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    dates.reverse()  # Oldest first
    
    # Get prediction counts for each day
    values = []
    for date_str in dates:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        next_date = date + datetime.timedelta(days=1)
        
        count = mongo['LeafSpec'].predictions.count_documents({
            "timestamp": {"$gte": date, "$lt": next_date}
        })
        values.append(count)
    
    return jsonify({
        "labels": dates,
        "values": values
    })

@admin_bp.route('/api/species-accuracy-data')
@admin_required
def species_accuracy_data():
    """Get species recognition accuracy data"""
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
        },
        {
            "$limit": 15  # Top 15 most predicted species
        }
    ]
    
    results = list(mongo['LeafSpec'].predictions.aggregate(pipeline))
    
    return jsonify({
        "labels": [item["_id"] for item in results],
        "values": [item["avg_confidence"] for item in results]
    })

@admin_bp.route('/api/api-usage-data')
@admin_required
def api_usage_data():
    """Get API usage data for the last 7 days"""
    days = 7
    
    # Generate dates
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [(today - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    dates.reverse()  # Oldest first
    
    # Get counts for each day
    predictions = []
    registrations = []
    feedback = []
    
    for date_str in dates:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        next_date = date + datetime.timedelta(days=1)
        
        # Predictions
        pred_count = mongo['LeafSpec'].predictions.count_documents({
            "timestamp": {"$gte": date, "$lt": next_date}
        })
        predictions.append(pred_count)
        
        # User registrations
        reg_count = mongo['LeafSpec'].users.count_documents({
            "registration_date": {"$gte": date, "$lt": next_date}
        })
        registrations.append(reg_count)
        
        # Feedback
        fb_count = mongo['LeafSpec'].feedback.count_documents({
            "timestamp": {"$gte": date, "$lt": next_date}
        })
        feedback.append(fb_count)
    
    return jsonify({
        "dates": dates,
        "predictions": predictions,
        "registrations": registrations,
        "feedback": feedback
    })

@admin_bp.route('/export-predictions')
@admin_required
def export_predictions():
    """Export predictions to CSV"""
    import csv
    from io import StringIO
    from flask import Response
    
    # Get all predictions
    predictions = list(mongo['LeafSpec'].predictions.find({}, {
        "image_data": 0  # Exclude image data
    }).sort("timestamp", -1))
    
    # Create CSV
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Species', 'Confidence', 'User', 'Timestamp'])
    
    # Write data
    for pred in predictions:
        writer.writerow([
            pred.get('species', ''),
            pred.get('confidence', ''),
            pred.get('user_email', 'Anonymous'),
            pred.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if pred.get('timestamp') else ''
        ])
    
    # Create response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=predictions.csv"}
    )
