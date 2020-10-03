import flask
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import sys
from zipfile import ZipFile
from io import BytesIO
import PIL
import PredictAndPostprocess


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
    result_json, result_image = PredictAndPostprocess.getPostprocessedResults(PredictAndPostprocess.getPrediction(floorplan), floorplan)
	
    in_memory = BytesIO()
    result_zip = ZipFile(in_memory, mode = 'w')
    result_zip.write(result_json)
    result_zip.write(result_image)
	
    result_zip.close()
    in_memory.seek(0)
    data = in_memory.read()
	
    with open('results.zip', 'wb') as out:
        out.write(data)

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

if __name__ == '__main__':
	app.run(debug = True)
