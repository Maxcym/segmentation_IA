from flask import Flask, render_template, redirect, url_for, flash, session, request
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from forms import LoginForm, RegistrationForm, ProfileUpdateForm
from models import User
from config import Config
from datetime import datetime
from segmentation import load_model, preprocess_image, predict, overlay_mask, compute_segmented_surface_from_mask
import os
from werkzeug.utils import secure_filename
import cv2
from models import db, SegmentationResult

# Initialisation de l'application Flask
app = Flask(__name__)
app.config.from_object(Config)

# Configuration des dossiers d'upload
UPLOAD_FOLDER = 'static/uploads/'
SEGMENTED_FOLDER = 'static/segmented/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER

# Création des dossiers si inexistants
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

db.init_app(app)

# Configuration de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return redirect(url_for('login'))

# Classe pour la gestion de l'admin
class AdminModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated and getattr(current_user, 'is_admin', False)

    def inaccessible_callback(self, name, **kwargs):
        flash("Accès refusé ! Vous devez être administrateur.", "danger")
        return redirect(url_for('login'))

# Ajout de l'interface d'administration Flask-Admin
admin = Admin(app, name='Admin Panel', template_mode='bootstrap3')
admin.add_view(AdminModelView(User, db.session))
admin.add_view(AdminModelView(SegmentationResult, db.session))


# Route d'inscription avec nouveaux champs
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        existing_email = User.query.filter_by(email=form.email.data).first()
        if existing_email:
            flash('Cet email est déjà utilisé.', 'danger')
        else:
            birthdate = datetime.strptime(str(form.birthdate.data), "%Y-%m-%d").date()
            new_user = User(
                firstname=form.firstname.data,
                lastname=form.lastname.data,
                profession=form.profession.data,
                birthdate=birthdate,
                email=form.email.data,
                is_admin=False
            )
            new_user.set_password(form.password.data)
            db.session.add(new_user)
            db.session.commit()
            flash('Inscription réussie ! Vous pouvez maintenant vous connecter.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html', form=form)

# Route de connexion
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Connexion réussie !', 'success')
            return redirect(url_for('dashboard'))
        flash('Email ou mot de passe incorrect.', 'danger')
    return render_template('login.html', form=form)

# Page protégée (Tableau de bord)
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

# Route pour choisir un modèle de segmentation
@app.route('/segmentation', methods=['GET', 'POST'])
@login_required
def segmentation():
    segmented_image = None
    original_image = None

    if request.method == 'POST':
        model_name = request.form['model']  # Récupération du modèle sélectionné par l'utilisateur
        file = request.files['image']

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            original_image = filename

            # Charger dynamiquement le modèle en fonction de la sélection de l'utilisateur
            model = load_model(model_name)
            image_tensor, original_img = preprocess_image(filepath)
            predicted_mask = predict(model, image_tensor)
            overlay_image = overlay_mask(original_img, predicted_mask)

            segmented_filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            segmented_filepath = os.path.join(app.config['SEGMENTED_FOLDER'], segmented_filename)
            cv2.imwrite(segmented_filepath, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

            # Sauvegarde des résultats en base de données
            new_result = SegmentationResult(
                user_id=current_user.id,
                original_image=filename,
                segmented_image=segmented_filename,
                model_used=model_name
            )
            db.session.add(new_result)
            db.session.commit()

            segmentation_result = new_result

            return render_template(
                'segmentation.html',
                segmented_image=segmented_filename,
                original_image=original_image,
                segmentation_result = segmentation_result
            )

    return render_template('segmentation.html', segmented_image=segmented_image, original_image=original_image)

# Route pour afficher les résultats de la segmentation
@app.route('/results')
@login_required
def results():
    results = SegmentationResult.query.filter_by(user_id=current_user.id).order_by(SegmentationResult.timestamp.desc()).all()
    return render_template('results.html', results=results)


@app.route('/delete_result/<int:result_id>', methods=['POST'])
@login_required
def delete_result(result_id):
    result = SegmentationResult.query.get(result_id)

    if result and result.user_id == current_user.id:  # Vérifier que l'utilisateur a bien accès à cet élément
        # Supprimer les fichiers images associés
        try:
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], result.original_image)
            segmented_path = os.path.join(app.config['SEGMENTED_FOLDER'], result.segmented_image)

            if os.path.exists(original_path):
                os.remove(original_path)
            if os.path.exists(segmented_path):
                os.remove(segmented_path)
        except Exception as e:
            flash(f"Erreur lors de la suppression des fichiers : {str(e)}", "danger")

        # Supprimer l'entrée de la base de données
        db.session.delete(result)
        db.session.commit()
        flash("Résultat supprimé avec succès.", "success")
    else:
        flash("Résultat introuvable ou accès non autorisé.", "danger")

    return redirect(url_for('results'))


# Route pour modifier le profil utilisateur
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ProfileUpdateForm(obj=current_user)
    if form.validate_on_submit():
        current_user.firstname = form.firstname.data
        current_user.lastname = form.lastname.data
        current_user.profession = form.profession.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Profil mis à jour avec succès.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('profile.html', form=form)


@app.route('/statistiques/<int:result_id>')
@login_required
def statistics(result_id):
    result = SegmentationResult.query.get(result_id)

    if not result or result.user_id != current_user.id:
        flash("Accès refusé ou résultat introuvable.", "danger")
        return redirect(url_for('results'))

    # Charger l'image segmentée et convertir en masque
    segmented_filepath = os.path.join(app.config['SEGMENTED_FOLDER'], result.segmented_image)
    segmented_mask = cv2.imread(segmented_filepath, cv2.IMREAD_GRAYSCALE) / 255.0

    segmented_pixel_count, segmented_surface_mm2 = compute_segmented_surface_from_mask(segmented_mask)

    cr_mesures_statistiques = f"L'image segmentée contient {segmented_pixel_count} pixels segmentés. La surface réelle estimée est de {segmented_surface_mm2:.2f} mm²."

    return render_template(
        'statistiques.html',
        original_image=result.original_image,
        segmented_image=result.segmented_image,
        model_used=result.model_used,
        timestamp=result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        segmented_pixel_count=segmented_pixel_count,
        segmented_surface_mm2=segmented_surface_mm2,
        cr_mesures_statistiques=cr_mesures_statistiques
    )


# Route de déconnexion
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('_flashes', None)
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('login'))

# Lancement de l'application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        email = "admin@gmail.com"  # Remplace par l'email de l'admin
        password = "admin123"  #

        existing_admin = User.query.filter_by(email=email).first()

        if existing_admin:
            print("Un administrateur avec cet email existe déjà.")
        else:
            admin_user = User(
                firstname="Admin",
                lastname="User",
                email=email,
                profession="Administrateur",  # Ajout d'une valeur par défaut
                birthdate=datetime.strptime("2000-01-01", "%Y-%m-%d").date(),  # Date de naissance par défaut
                is_admin=True
            )

            admin_user.set_password(password)
            db.session.add(admin_user)
            db.session.commit()
            print("Administrateur créé avec succès.")

    app.run(debug=True)
