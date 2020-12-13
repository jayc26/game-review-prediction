from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,PasswordField,BooleanField,IntegerField,FileField
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_bootstrap import Bootstrap
from wtforms.validators import DataRequired,Length,Email,EqualTo
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

images = UploadSet('images', IMAGES)
class RegF(FlaskForm):
    u=StringField('Name',validators=[DataRequired()])
    d=StringField('Description', validators=[DataRequired()])
    r=IntegerField('Room Number',validators=[DataRequired()])
    s=IntegerField('Salary',validators=[DataRequired()])
    i=FileField("Image",validators=[DataRequired()])
    submit= SubmitField('Add Data')

class ImgF(FlaskForm):
    
    u=StringField('Enter Name to search an Image',validators=[DataRequired()])
    submit= SubmitField('Search')

class IdF(FlaskForm):
    
    u=StringField('Enter ID to search an Image an Caption',validators=[DataRequired()])
    submit= SubmitField('Search')


class SalF(FlaskForm):
    
    u=IntegerField('Enter Salary',validators=[DataRequired()])
    submit= SubmitField('Search')
class UsalF(FlaskForm):
    
    u=StringField('Enter Name',validators=[DataRequired()])
    s=StringField('Enter Salary',validators=[DataRequired()])

    submit= SubmitField('Update')

class UkeyF(FlaskForm):
    
    u=StringField('Enter Name',validators=[DataRequired()])
    s=StringField('Enter Keyword',validators=[DataRequired()])

    submit= SubmitField('Update')

class USalF(FlaskForm):
    
    u=StringField('Enter Name',validators=[DataRequired()])
    s=StringField('Enter New Salary',validators=[DataRequired()])

    submit= SubmitField('Update')

class DelF(FlaskForm):
    
    u=StringField('Enter Name',validators=[DataRequired()])
    

    submit= SubmitField('Update')

class UImF(FlaskForm):
    
    u=StringField('Enter Name',validators=[DataRequired()])
    #s=FileField('Enter New Image',validators=[DataRequired()])
    upload = FileField('image', validators=[
        FileRequired(),
        FileAllowed(images, 'Images only!')
    ])

    submit= SubmitField('Update')



