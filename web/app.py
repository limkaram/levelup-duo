from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import subprocess
import utils

app = Flask(__name__)
PROJ_ROOT_PATH = utils.get_project_root()
TEMP_INPUT_FILENAME = ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object-detection')
def object_detection():
    return render_template('objectDetection.html')

@app.route('/sentiment-analysis')
def sentiment_analysis():
    return 'sentiment analysis page'


@app.route('/detect', methods=['POST'])
def detect():
    if not request.method == 'POST':
        return

    global TEMP_INPUT_FILENAME

    f = request.files['file']
    input_filepath = os.path.join(PROJ_ROOT_PATH, 'web', 'uploads', secure_filename(f.filename))
    input_filename = os.path.basename(input_filepath)
    TEMP_INPUT_FILENAME = input_filename
    f.save(input_filepath)

    model_path = os.path.join(PROJ_ROOT_PATH, 'yolov5', 'runs', 'train', 'limkaram_yolov5s_img416_batch16_epoch400', 'weights', 'best.pt')
    script_path = os.path.join(PROJ_ROOT_PATH, 'yolov5', 'detect.py')
    output_dirpath = os.path.join(PROJ_ROOT_PATH, 'web', 'outputs', input_filename)
    output_filepath = os.path.join(output_dirpath, input_filename)
    print('[model_path]', model_path)
    print('[script_path]', script_path)
    print('[output_filepath]', output_filepath)

    command = ['python', script_path, '--weights', model_path, '--img', '416', '--conf', '0.5', '--source', input_filepath, '--name', output_dirpath]
    subprocess.run(command)

    return render_template('detectComplete.html', output_filepath=output_filepath)


@app.route('/return-file', methods=['GET'])
def return_file():
    output_filepath = os.path.join(PROJ_ROOT_PATH, 'web', 'outputs', TEMP_INPUT_FILENAME, TEMP_INPUT_FILENAME)
    print(output_filepath)

    try:
        return send_file(output_filepath, attachment_filename=TEMP_INPUT_FILENAME)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
