import flask
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import sys
import zipfile
from io import BytesIO
import PIL
import PredictAndPostprocess
import json

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

app = flask.Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.static_folder = 'static'

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

@app.route('/predict')
@cross_origin()
def waiting_prediction():
    

    floorplan_path = os.path.join(app.config['UPLOAD_FOLDER'], "floorplan.png")
    floorplan = PIL.Image.open(floorplan_path).convert('L').convert('RGB')
    result_image, result_json = PredictAndPostprocess.getPostprocessedResults(PredictAndPostprocess.getPrediction(floorplan), floorplan)
	
    result_image.save(os.path.join(app.config['UPLOAD_FOLDER'], "../results/result_image.png"))
    
    with open("results/result_json.json", 'w') as json_path:
        json.dump(result_json, json_path)

    zipf = zipfile.ZipFile('./templates/results.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('results/', zipf)
    zipf.close()
    
    print("RESULT IS READY")

    #return flask.redirect(flask.url_for('show_download_page'))
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}

@app.route('/download')
def show_download_page():
    
    print('REDIRECTED TO DOWNLOAD')
    
    return flask.render_template('download_result.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if flask.request.method == 'POST':
		file = flask.request.files['file']
		filename = "floorplan.png"
		print(filename, file = sys.stdout)
		print(APP_ROOT, file = sys.stdout)
		print(UPLOAD_FOLDER, file = sys.stdout)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return flask.render_template('wait.html')

@app.route('/results.zip', methods=['GET', 'POST'])
def download():
    uploads = os.path.join(app.config['UPLOAD_FOLDER'], '../templates')
    return flask.send_from_directory(directory=uploads, filename='results.zip')

if __name__ == '__main__':
	app.run(debug = True)
