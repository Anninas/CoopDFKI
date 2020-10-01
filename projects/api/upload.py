import flask
from werkzeug.utils import secure_filename
import os
import sys
#import PredictPostprocess.py


app = flask.Flask(__name__)

APP_ROOT = os.path.dirname(__file__)
UPLOAD_PATH = 'upload_pictures'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_PATH)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global result_image
global result_json

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def show_upload_page():
   return flask.render_template('mainpage.html')

   
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if flask.request.method == 'POST':
      file = flask.request.files['file']
      filename = secure_filename(file.filename)
      print(filename, file = sys.stdout)
      print(APP_ROOT, file = sys.stdout)
      print(UPLOAD_FOLDER, file = sys.stdout)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	  #result_json, result_image = PredictPostprocess.postprocess(PredictPostprocess.predict)
	  #return flask.redirect(flask.url_for('waiting for prediction'))
      return flask.redirect(flask.url_for('uploaded_file', filename=filename))
      #return 'file uploaded successfully'

if __name__ == '__main__':
   app.run(debug = True)
   