import os
import shutil
import csv
import sys
from flask import Flask,render_template, url_for, flash, redirect, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
# from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_bootstrap import Bootstrap
from wtforms import StringField, IntegerField, SubmitField, SelectField,FileField, TextAreaField
from wtforms.validators import DataRequired
#from forms import RegF,ImgF,SalF,UkeyF,USalF,UImF,DelF,IdF
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

UPLOAD_FOLDER = './upload'

app = Flask(__name__ , static_url_path="/static")
bootstrap = Bootstrap(app)

# Configurations
app.config['SECRET_KEY'] = 'blah blah blah blah'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#db = SQLAlchemy(app)


class NameForm(FlaskForm):

	name = TextAreaField('',render_kw={"rows": 20, "cols": 8, "align":"center"})
	submit = SubmitField('Submit', render_kw={"margin-left":"500px"})

# ROUTES!
@app.route('/',methods=['GET','POST'])
def index():
	form = NameForm()
	if form.validate_on_submit():
		name = form.name.data
		#token = RegexpTokenizer(r'[a-zA-Z0-9]+')
		#cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
		filename="Model/cvrc.sav"
		loaded_model = pickle.load(open(filename, 'rb'))
		cv = pickle.load(open('Model/cv', 'rb'))
		name=cv.transform([name])
		result = loaded_model.predict(name)
		print(result)
		if result[0] <= 4.0:
			rew="Bad Review"
		elif result[0] == 5.0:
			rew="Neutral Review"
		elif result[0] > 5.0:
			rew="Good Review"
		
		return render_template('index.html',form=form,name=result[0], rew=rew)
	return render_template('index.html',form=form,name=None)
	
# @app.route('/register', methods=['GET','POST'])
# def regi():
# 	reg= RegF()
# 	if reg.validate_on_submit():
# 		name=reg.u.data
# 		flash(f'Account created for {reg.u.data}','success')
# 		return redirect(url_for('index'))
# 	return render_template('register.html',title="Register",form=reg)



# @app.route('/id', methods=['GET','POST'])
# def im():
# 	im= IdF()
# 	if im.validate_on_submit():
		
    
# 		name=im.u.data
# 		data = pd.read_csv("names-1.csv")
# 		print(data)
# 		i=data[data.ID==name].Picture.values
# 		j=i=data[data.ID==name].Caption.values
# 		print(i)
# 		if i:

# 			return render_template('id.html',form=im,Name=name, Cap=j,Image=i[0])
# 		else:
# 			return render_template('id.html',form=im,Image="None")
			
# 	return render_template('index.html',title="Image",form=im)
# @app.route('/salary', methods=['GET','POST'])

# @app.route('/image', methods=['GET','POST'])
# def idm():
# 	im= ImgF()
# 	if im.validate_on_submit():
		
    
# 		name=im.u.data
# 		data = pd.read_csv("names-1.csv")
# 		print(data)
# 		i=data[data.Picture==name].Picture.values
# 		print(i)
# 		if i:

# 			return render_template('image.html',form=im,Name=name,Image=i[0])
# 		else:
# 			return render_template('image.html',form=im,Image="None")
			
# 	return render_template('index.html',title="Image",form=im)
# @app.route('/salary', methods=['GET','POST'])



# def sal():
# 	sal= SalF()
# 	if sal.validate_on_submit():
# 		s=sal.u.data
# 		d=data[data.Salary < s].Picture.values
# 		if d:

# 			return render_template('salary.html',form=sal,Image=d)
# 		else:
# 			return render_template('salary.html',form=sal,Image="None")
# 	return render_template('index.html',title="Salary",form=sal)

# @app.route('/upkey', methods=['GET','POST'])
# def ukey():
# 	uk= UkeyF()
# 	if uk.validate_on_submit():
# 		name=uk.u.data
# 		k=uk.s.data
# 		data.at[data.Name==name,"Caption"] = k
# 		data.to_csv("names-1.csv",index=False)
# 		dat = pd.read_csv("names-1.csv")
# 		i=dat[dat.Name==name].Picture.values
# 		j=dat[dat.Name==name].Caption.values
# 		print(i,j)

# 		return render_template('upkey.html',form=uk,Name=name,Image=i, cap=j)
# 	return render_template('upkey.html',title="Salary",form=uk)

# @app.route('/upsal', methods=['GET','POST'])
# def usal():
# 	us= USalF()
# 	if us.validate_on_submit():
# 		name=us.u.data
# 		k=us.s.data
# 		data.at[data.Name==name,"Salary"] = k
# 		data.to_csv("people.csv",index=False)
		
# 		return render_template('upsal.html',form=us,Name=name,Kw=k)
# 	return render_template('upsal.html',title="Salary",form=us)

# @app.route('/del', methods=['GET','POST'])
# def delr():
# 	us= DelF()
# 	if us.validate_on_submit():
# 		name=us.u.data
# 		print(name)
# 		#data.drop([data.Name==name])
# 		b=data[data.Name != name]
# 		print(data)
# 		print(b)
# 		data1=b
# 		#data.at[data.Name==name,"Salary"] = k
# 		data1.to_csv("people.csv",index=False)
# 		return render_template('del.html',form=us,Name=name)
# 	return render_template('del.html',title="Salary",form=us)

# @app.route('/upload', methods=['GET','POST'])
# def uim():
# 	us= UImF()
# 	if us.validate_on_submit():
# 		name= us.u.data
		


		
# 		return render_template('upsal.html',form=us,Name=name)
# 	return render_template('upsal.html',title="Salary",form=us)


@app.route('/help')
def help():
	text_list = []
	# Python Version
	text_list.append({
		'label':'Python Version',
		'value':str(sys.version)})
	# os.path.abspath(os.path.dirname(__file__))
	text_list.append({
		'label':'os.path.abspath(os.path.dirname(__file__))',
		'value':str(os.path.abspath(os.path.dirname(__file__)))
		})
	# OS Current Working Directory
	text_list.append({
		'label':'OS CWD',
		'value':str(os.getcwd())})
	# OS CWD Contents
	label = 'OS CWD Contents'
	value = ''
	text_list.append({
		'label':label,
		'value':value})
	return render_template('help.html',text_list=text_list,title='help')

@app.errorhandler(404)
@app.route("/error404")
def page_not_found(error):
	return render_template('404.html',title='404')

@app.errorhandler(500)
@app.route("/error500")
def requests_error(error):
	return render_template('500.html',title='500')
if __name__ == '__main__':

	#app.run(debug=True)

	port = int(os.getenv('PORT', '3000'))
	app.run(host='0.0.0.0', port=port, debug=True)