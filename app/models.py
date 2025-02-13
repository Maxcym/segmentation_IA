from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(150), nullable=False)
    lastname = db.Column(db.String(150), nullable=False)
    profession = db.Column(db.String(150), nullable=False)
    birthdate = db.Column(db.Date, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SegmentationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Lien avec l'utilisateur
    original_image = db.Column(db.String(255), nullable=False)  # Chemin de l'image originale
    segmented_image = db.Column(db.String(255), nullable=False)  # Chemin de l'image segmentée
    model_used = db.Column(db.String(50), nullable=False)  # Modèle utilisé
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Date de sauvegarde

    user = db.relationship('User', backref='segmentations', lazy=True)  # Relation avec User