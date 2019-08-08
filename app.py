#!/usr/bin/env python3
from flask_pymongo import PyMongo
from flask import Flask, render_template,request,redirect,url_for,jsonify, session

from pymongo import MongoClient
import os
import pandas as pd
import numpy as np
import random, json
import zipfile as zf
import shutil
import re
import seaborn as sns
from flask_sqlalchemy import SQLAlchemy
from fastai.vision import get_transforms
from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
from glob2 import glob
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/predict/<filename>', methods=['POST','GET'])
def predict(filename):
    cwd = os.getcwd()
    img = './data/test/'+filename
    print("This is" ,img)
    path = Path(os.getcwd())/"data"
    tfms = get_transforms(do_flip=True, flip_vert=True)
    data = ImageDataBunch.from_folder(
    path, test="test", ds_tfms=tfms, bs=16, num_workers=0)
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    model = learn.load("trash")
    img1 = open_image(img)
    predict = model.predict(img1)
    max = float(predict[2][0])
    pos = 0
    for i in range(len(predict[2])):
        if float(predict[2][i]) > max:
            max = float(predict[2][i])
            pos = i
    print("Waste is classified as:", data.classes[pos])
    return jsonify(data.classes[pos])
    

if __name__ == "__main__":
    app.run(port=5000,debug=True,use_reloader=True)