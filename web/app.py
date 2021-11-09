from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import subprocess
import customUtils
import torch
import numpy as np
from useModels.kcelectra import load_sa_model, infer


app = Flask(__name__)
PROJ_ROOT_PATH = customUtils.get_project_root()
TEMP_INPUT_FILENAME = ''
yolo_root_path = r'C:\Users\LIMKARAM\PycharmProjects\yolov5'
od_model_path = r'C:\Users\LIMKARAM\PycharmProjects\yolov5\runs\train\limkaram_yolov5s_img416_batch16_epoch400\weights\best.pt'
od_model = torch.hub.load(yolo_root_path, 'custom', path=od_model_path, source='local')  # local repo

sa_model_path = r'C:\Users\LIMKARAM\PycharmProjects\kcelectra\epoch4-val_acc0.9195.ckpt'
sa_model = load_sa_model(sa_model_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/object-detection')
def object_detection():
    return render_template('objectDetection.html')


@app.route('/sentiment-analysis')
def sentiment_analysis():
    return render_template('sentimentAnalysis.html')


# TODO : 작성 필요
@app.route('/sentiment-analysis/sa-predict', methods=['POST'])
def sa_predict():
    input = request.form['text']
    pred = infer(sa_model, input)
    pred[0], pred[1] = f'{np.round(pred[0] * 100, 1):.1f}', f'{np.round(pred[1] * 100, 1):.1f}'
    print('입력 텍스트 : ', input)
    print('모델 예측 : ', pred)

    return render_template('saComplete.html', input=input, pred=pred)


@app.route('/object-detection/od-fileUpload', methods=['POST'])
def od_upload_and_predict():
    if not request.method == 'POST':
        return

    global TEMP_INPUT_FILENAME

    f = request.files['file']
    input_filename = secure_filename(f.filename)
    input_filepath = os.path.join(PROJ_ROOT_PATH, 'web', 'uploads', input_filename)
    TEMP_INPUT_FILENAME = input_filename
    f.save(input_filepath)

    # TODO : path config로 변환
    model_path = r'C:\Users\LIMKARAM\PycharmProjects\yolov5\runs\train\limkaram_yolov5s_img416_batch16_epoch400\weights\best.pt'
    script_path = r'C:\Users\LIMKARAM\PycharmProjects\yolov5\detect.py'
    output_dirpath = os.path.join(PROJ_ROOT_PATH, 'web', 'outputs', input_filename)
    output_filepath = os.path.join(output_dirpath, input_filename)
    print('[model_path]', model_path)
    print('[script_path]', script_path)
    print('[input_filepath]', input_filepath)
    print('[output_filepath]', output_filepath)

    command = ['python', script_path, '--weights', model_path, '--img', '416', '--conf', '0.5', '--source', input_filepath, '--name', output_dirpath]
    subprocess.run(command)

    return render_template('detectComplete.html', output_filepath=output_filepath)


@app.route('/object-detection/api', methods=['POST'])
def ob_api():
    if not request.method == 'POST':
        return

    global model

    f = request.files['img_file']
    save_path = os.path.join(PROJ_ROOT_PATH, 'web', 'uploads', secure_filename(f.filename))
    f.save(save_path)
    print(save_path)
    result = od_model(save_path)
    x, y, w, h, score, class_num, class_name = result.pandas().xywh[0].loc[0]
    print(x, y, w, h, score, class_num, class_name)

    return jsonify({'x_center': str(x),
                      'y_center': str(y),
                      'width': str(w),
                      'height': str(h),
                      'confidence_score': str(score),
                      'class_num': str(class_num),
                      'class_name': str(class_name)})


@app.route('/object-detection/od-fileUpload/return-file', methods=['GET'])
def return_file():
    output_filepath = os.path.join(PROJ_ROOT_PATH, 'web', 'outputs', 'videos', TEMP_INPUT_FILENAME, TEMP_INPUT_FILENAME)
    print(output_filepath)

    try:
        return send_file(output_filepath, attachment_filename=TEMP_INPUT_FILENAME)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
