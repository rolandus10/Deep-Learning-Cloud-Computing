import os
from app import app
from matplotlib import pyplot as plt
from keras.applications.vgg16 import preprocess_input
#from machineLearning import Cal_SIFT,Cal_ORB,Cal_LBP,bruteForceMatching,bhatta
from CodeDeepLearning import euclidianDistance,chi2_distance,search,rappelPrecision,combiner
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
import csv
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from keras.applications.xception import preprocess_input

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
calcul_distance = 1
top = 20

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/set_Euclidean",methods=["GET","POST"])
def set_Euclidean():
    global calcul_distance
    calcul_distance = 1
    return redirect(url_for('upload_form'))

@app.route("/set_ChiSquare",methods=["GET","POST"])
def set_ChiSquare():
    global calcul_distance
    calcul_distance = 2
    return redirect(url_for('upload_form'))

@app.route("/set_Bhatta",methods=["GET","POST"])
def set_Bhatta():
    global calcul_distance
    calcul_distance = 2
    return redirect(url_for('upload_form'))

@app.route("/set_BruteForceMatching",methods=["GET","POST"])
def set_BruteForceMatching():
    global calcul_distance
    calcul_distance = 3
    return redirect(url_for('upload_form'))

@app.route("/get_top20",methods=["GET","POST"])
def get_top20():
    global top
    top = 20
    return redirect(url_for('upload_form'))

@app.route("/get_top50",methods=["GET","POST"])
def get_top50():
    global top
    top = 50
    return redirect(url_for('upload_form'))

@app.route("/get_top100",methods=["GET","POST"])
def get_top100():
    global top
    top = 100
    return redirect(url_for('upload_form'))

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    global top
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], file_names[0])
     
    modelInception = load_model('modelInceptionv3.h5')
    modelvgg = load_model('modelvgg16.h5')
    image = load_img(img1_path, target_size = (299, 299))
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image) #fonction importer
    # predict the probability across all output classes
    #feature = modelvgg.predict(image)
    #feature = feature[0]
    feature= combiner(image, modelvgg,modelInception)

    courbe1=os.path.splitext(os.path.basename(file_names[0]))[0]+'z'

   
    #feature= image_entree(image,modelvgg)
    voisins = search(feature,"features/Concat.csv",calcul_distance,top)
    Value_RP=rappelPrecision(voisins)
    X=Value_RP[0]
    Y=Value_RP[1]
    plt.clf()
    plt.plot(X[1:],Y[1:], label="Rappel/Precision")
    plt.xlabel('Rappel')
    plt.ylabel('Précison')
    plt.title("R/P")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'],courbe1))

    return render_template('upload.html', filenames=voisins, courbe=courbe1+'.png',imageRequete=file_names[0])

@app.route('/display/<filename>')
def display_image2(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='GHIM/' + filename), code=301)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000)
