from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, IntegerField, SubmitField, DateField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange

class RegistrationForm(FlaskForm):
    firstname = StringField('Prénom', validators=[DataRequired(), Length(min=2, max=150)])
    lastname = StringField('Nom', validators=[DataRequired(), Length(min=2, max=150)])
    profession = StringField('Profession', validators=[DataRequired(), Length(min=2, max=150)])
    birthdate = DateField('Date de naissance', format='%Y-%m-%d', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    password = PasswordField('Mot de passe', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirmer le mot de passe', validators=[DataRequired(), EqualTo('password', message="Les mots de passe ne correspondent pas")])
    submit = SubmitField("S'inscrire")

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired()])
    password = PasswordField('Mot de passe', validators=[DataRequired()])
    submit = SubmitField('Se connecter')

class ProfileUpdateForm(FlaskForm):
    firstname = StringField('Prénom', validators=[DataRequired()])
    lastname = StringField('Nom', validators=[DataRequired()])
    profession = StringField('Profession', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    submit = SubmitField('Mettre à jour')
