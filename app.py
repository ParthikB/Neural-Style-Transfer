from flask import Flask, render_template, request, redirect
import numpy as np
import joblib
import os, time, random
from PIL import Image
import numpy as np

from __helpers__ import *

app = Flask(__name__)

# Creating a user_file folder if not present in the working dir
if 'temp_user_files' not in os.listdir():
	os.mkdir('temp_user_files')
# temp_user_file : It'll contain the images uploaded by the user temporarily

app.config["IMAGE_UPLOADS"] = os.path.join(os.getcwd(), 'temp_user_files')
home_dir = os.getcwd()

# Defining the Home page
@app.route('/')
def home():
	return render_template('index.html')


# Defining the Prediction page
@app.route('/', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST' and request.form:
		# Reading the POST request
		content_img = request.files['img1name']
		style_img   = request.files['img2name']

		# Saving the Image file in local env
		content_path = os.path.join(app.config["IMAGE_UPLOADS"], 'content_img.jpg')
		style_path   = os.path.join(app.config["IMAGE_UPLOADS"], 'style_img.jpg')

		content_img.save(content_path)
		style_img.save(style_path)

		generatedImage, losses = runStyleTransfer(content_path,
								style_path,
								iterations     = 2,
								SAVE_EVERY     = 0,
								contentWeight  = 1,
								styleWeight    = 0.8,
								output_dirName = 'static',
								save_stats     = False)


		# Deleting the file from the database
		os.remove(content_path)
		os.remove(style_path)
		
		# Switching back to the main dir because Current dir is changed internally
		os.chdir(home_dir)
		
	return render_template('index.html')


if __name__ == '__main__':
	app.run(port=int(random.random()*10000), debug=True)