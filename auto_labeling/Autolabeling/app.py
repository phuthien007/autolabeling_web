import glob
import shutil
from flask import Flask, request, make_response, render_template
from flask_cors import CORS
import zipfile
import os
from os.path import basename
from training_class import train
from detect_coordinates import *
app = Flask(__name__, static_url_path='/static')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


def unzip(path_to_unzip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_unzip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def zip(path_to_zip_file, dirName):
    # create a ZipFile object
    with zipfile.ZipFile(path_to_zip_file, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(dirName):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, basename(filePath))

def deleteAllFile(dir):
    # Deleting an non-empty folder
    dir_path =dir
    shutil.rmtree(dir_path, ignore_errors=True)
    print("Deleted '%s' directory successfully" % dir_path)


@app.route('/api/upload-data-classifier', methods=['POST'])
def getDataClassifier():
    if request.method == 'POST':
        if (len(request.access_route) < 2):
            data = request.files.get('file')
            data.save(dst='./static/upload_data/data.zip')
            print(data)
            unzip('./static/upload_data/data.zip', './training_class/')
            return make_response({"message":"success"}), 200
        else:
            return make_response({'message': 'error'}), 400

@app.route('/api/upload-data-boundingbox', methods=['POST'])
def getDataBoundingbox():
    if request.method == 'POST':
        if (len(request.access_route) < 2):
            data = request.files.get('file')
            data.save(dst='./static/upload_data/images.zip')
            print(data)
            unzip('./static/upload_data/images.zip', './datasets/')
            return make_response({"message":"success"}), 200
        else:
            return make_response({'message': 'error'}), 400
@app.route('/api/upload-model', methods=['POST'])
def getModel():
    if request.method == 'POST':
        if (len(request.access_route) < 2):
            data = request.files.get('file')
            data.save(dst='./static/upload_data/best.pt')
            print(data)
            resetPretrain(False)
            return make_response({"message":"success"}), 200
        else:
            return make_response({'message': 'error'}), 400


@app.route('/api/get-result', methods=['POST'])
def processAndSendFile():
    if(len(request.access_route) <2):

        train.process_train('training_class\\data\\')
        process_main('training_class\\data\\')
        zip('./static/images.zip', './datasets/images')
        if pretrained == False:
            os.remove('./upload_data/best.pt')
            resetPretrain(True)
        os.remove(os.path.join('training_class\\data\\', 'my_weight.pt'))
        os.remove('./static/upload_data/data.zip')
        os.remove('./static/upload_data/images.zip')
        deleteAllFile('./datasets/images')
        deleteAllFile('./training_class/data')
        return  make_response({'message': os.path.join(request.root_url,'static/images.zip') }), 200
    else:
        return make_response({'message':'error'}), 400

# @app.route('/')
# def home():
#     return render_template('./template/index.html')

if __name__ == '__main__':
    app.run()
    os.remove('./static/images.zip')
    # deleteAllFile('.\\training_class\\data')
    # deleteAllFile('/datasets/images')

