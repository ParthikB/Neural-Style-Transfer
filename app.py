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

# To DISABLE Caching
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# Defining the Home page
@app.route('/', methods=['GET', 'POST'])
def home():
	return render_template('index.html')


# Defining the Prediction page
@app.route('/output', methods=['GET', 'POST'])
def output():

	if request.method == 'POST':
		if request.form['submit'] == 'Do the magic!':

	# 		# Reading the POST request
			content_img = request.files['img1name']
			style_img   = request.files['img2name']
			iterations  = int(request.form['iterations'])
			print('Iterations :',iterations)

	# 		# Saving the Image file in local env
			content_path = os.path.join(app.config["IMAGE_UPLOADS"], 'content_img.jpg')
			style_path   = os.path.join(app.config["IMAGE_UPLOADS"], 'style_img.jpg')
			content_img.save(content_path)
			style_img.save(style_path)

	# 		# Running Style Transfer
			print('[INFO] Style Transfer Intializing..')
			generatedImage, losses = runStyleTransfer(content_path,
													style_path,
													iterations     = iterations,
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
			print('[INFO] Output generated.')
			return render_template('output.html')
	
		print('[INFO] Rendering Output')
		return render_template('index.html')
		


# if __name__ == '__main__':
# 	app.run(port=5555, debug=True)

# int(random.random()*10000)
