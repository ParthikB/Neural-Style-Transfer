import os

# To suppress all the Tensorflow warning while importing
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import keras.backend as K
from keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras import models
from keras.models import Model
from keras.preprocessing import image

import requests
from PIL import Image
import numpy as np
import cv2
import urllib
from tqdm import tqdm

if tf.__version__[0] == '1':
	tf.enable_eager_execution()
# print("Eager Execution Initialized:",tf.executing_eagerly())


# Starting session to download Images from URL if fed.
s = requests.Session()
s.proxies = {"http": "http://61.233.25.166:80"}
r = s.get("http://www.google.com")


# Defining the Feature Layers we need respectively
styleLayers = ['block1_conv2', 
               'block2_conv2', 
               'block3_conv3', 
               'block4_conv3', 
               'block5_conv3']

contentLayer = ['block3_conv2']


numContentLayers = len(contentLayer) # Number of Content Layers
numStyleLayers   = len(styleLayers)  # Number of Style Layers


# Defining Function to import model (VGG19)
def getModel():

  # Loading Model from tf
  model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  model.trainable = False # Freezing to parameters

  # Features of the Respective Layers
  contentFeatures = [model.get_layer(name).output for name in contentLayer]
  styleFeatures   = [model.get_layer(name).output for name in styleLayers]
  
  modelOutput = contentFeatures + styleFeatures

  return models.Model(model.input, modelOutput)


 # Defining GRAM MATRIX

def gram(x):

  # number of channels
  channels = int(x.shape[-1])
  
  # reshaping to channel first
  a = tf.reshape(x, [-1, channels])
  n = tf.shape(a)[0]
  
  # gram matrix
  gram = tf.matmul(a, a, transpose_a=True)
  
  return gram / tf.cast(n, tf.float32)

# Defining CONTENT COST
def contentCost(contentFeatures, generateFeatures):
  return tf.reduce_mean(tf.square(contentFeatures-generateFeatures))

# Defining STYLE COST
def styleCost(styleFeatures, generateFeatures):
  styleGram = gram(styleFeatures)
  return tf.reduce_mean(tf.square(styleGram - generateFeatures))


def getFeatures(content, style, model):
  
  # Defining the respective outputs from our model
  contentOutputs = model(content)
  styleOutputs = model(style)
  
  # Extracting out the different features from the model output
  contentFeatures = [contentFeature[0] for contentFeature in contentOutputs[numStyleLayers:]]
  styleFeatures = [styleFeature[0] for styleFeature in styleOutputs[:numStyleLayers]]

  return contentFeatures, styleFeatures


def loadImage(path_to_img):
  max_dim = 512
  if type(path_to_img) == str:
    img2show = Image.open(path_to_img)
  else:
    img2show = path_to_img

  # long = max(img2show.size)
  long = (img2show.size)

  scale = max_dim/long
  img = img2show.resize((round(img2show.size[0]*scale), round(img2show.size[1]*scale)), Image.ANTIALIAS)
  
  img = image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img2show, axis=0)
  return img2show, img
 

def urlToImage(url):
  resp = urllib.request.urlopen(url)
  img = np.asarray(bytearray(resp.read()), dtype='uint8')
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  img2show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  img = np.expand_dims(img2show, axis=0)
  return img2show, img
 

def inputImageAndPreprocess(path):
  # Loading the image and reshaping it according to VGG19 requirements.
  if path[:4]=='http':
    # print("Loading Image from Internet...")
    img2show, img = urlToImage(path)
  else:
    # print("Loading Image from Local...")
    img2show, img = loadImage(path)
  
  # Preprocessing the img according to VGG19 requirements
  img = tf.keras.applications.vgg19.preprocess_input(img)

  return img2show, img


# Deprocessing Image to save locally
def deprocessImage(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x


def totalLoss(model, lossWeights, generateImage, contentFeatures, styleFeatures):

  # Extracting the respective weights
  contentWeight, styleWeight = lossWeights

  # Extracting the generate image features from the model
  modelOutputs = model(generateImage)

  # Splitting the generate Features into different categories
  contentGenerateFeatures = modelOutputs[numStyleLayers:]
  styleGenerateFeatures   = modelOutputs[:numStyleLayers]

  # Initializing all costs with 0
  contentCostValue, styleCostValue = 0, 0

  # Defining partial weights
  contentWeightPerLayer = 1.0 / float(numContentLayers)
  styleWeightPerLayer = 1.0 / float(numStyleLayers)

  # Computing Content Cost
  for generateContent, combinationContent in zip(contentFeatures, contentGenerateFeatures):
    contentCostValue += contentWeightPerLayer * contentCost(combinationContent[0], generateContent)

  # Computing Style Cost for every layer
  for generateStyle, combinationStyle in zip(styleFeatures, styleGenerateFeatures):
    styleCostValue += styleWeightPerLayer * styleCost(combinationStyle[0], generateStyle)
  

  # Assigning the weights
  contentCostValue *= contentWeight
  styleCostValue *= styleWeight

  # Computing the Total Loss
  totalLossValue = contentCostValue + styleCostValue

  return totalLossValue, contentCostValue, styleCostValue


def computeGrads(config):
	with tf.GradientTape() as tape:
	  allLoss = totalLoss(**config)

	loss = allLoss[0]

	return tape.gradient(loss, config['generateImage']), allLoss


def extract_frames_out_of_the_video(vid_path):
  cam = cv2.VideoCapture(vid_path)

  currentframe = 1
  frames = []

  while(True):   
      # reading from frame 
      ret, frame = cam.read() 
      
      if ret:
          frame = np.expand_dims(frame, axis=0)
          frames.append(frame.astype('float32'))
          print(currentframe, end='\r') 
          currentframe += 1
      else: 
          break
  print('Frames generated..!')
  return frames


def skip_frame_every(fps_quality):
	if fps_quality == 'high':
	  skip_frame_every = 1
	elif fps_quality == 'medium':
	  skip_frame_every = 3
	elif fps_quality == 'low':
	  skip_frame_every = 6
	else:
	  skip_frame_every = 10

	return skip_frame_every


# Defining the MAIN TRAINING FUNCTION
def runStyleTransfer(contentPath,
                     stylePath,
                     iterations     = 1000,
                     SAVE_EVERY     = 0,
                     contentWeight  = 1e3,
                     styleWeight    = 1e-2,
                     output_dirName = None):
    
  # Importing the Model
  model = getModel()
  for layer in model.layers:
    layer.trainable = False

  if type(contentPath) == str:
    _ , contentImage = inputImageAndPreprocess(contentPath)
  else:
    contentImage = contentPath
    
  _ , styleImage = inputImageAndPreprocess(stylePath)
  
  # Extracting out the respective features from the model
  contentFeatures, styleFeatures = getFeatures(contentImage, styleImage, model)
  styleFeatures = [gram(styleFeature) for styleFeature in styleFeatures]

  # Creating the Generate Image
  generateImage = contentImage
  generateImage = tf.Variable(generateImage, dtype=tf.float32)

  # Defining the Adam Optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=5, epsilon=1e-3)

  # Storing the best Image and Loss
  bestLoss, bestImage = float('inf'), None

  # Zipping the Weights
  lossWeights = (contentWeight, styleWeight)
  
  # Defining the Config File
  config = {
    'model': model,
    'lossWeights': lossWeights,
    'generateImage': generateImage,
    'contentFeatures': contentFeatures,
    'styleFeatures': styleFeatures
    }
  
  
  normMeans = np.array([103.939, 116.779, 123.68])
  minVals = -normMeans
  maxVals = 255 - normMeans  

  # Creating Logs to use for Plotting Later
  # global contentCostLog, styleCostLog, totalCostLog 
  contentCostLog, styleCostLog, totalCostLog = [], [], []


  if output_dirName:
    PATH = os.path.join(os.curdir, 'outputs', output_dirName)
    if not os.path.isdir(PATH):
      os.mkdir(PATH)
    os.chdir(PATH)

  for iter in tqdm(range(iterations), leave=False):

    # Computing the Grads and Loss
    grads, allLoss = computeGrads(config)

    # Extracting different kinds of Losses
    loss, contentLoss, styleLoss = allLoss
    
    # Saving the respective losses in respective lists for plotting
    contentCostLog.append(contentLoss)
    styleCostLog.append(styleLoss)
    totalCostLog.append(loss)

    # Applying gradients to Generate Image
    optimizer.apply_gradients([(grads, generateImage)])

    # Clipping the values of Generate Image from (-255, 255)
    clipped = tf.clip_by_value(generateImage, minVals, maxVals)
    generateImage.assign(clipped)

    # Updating the Best Image and Loss
    if loss < bestLoss:
      bestLoss = loss
      bestImage = deprocessImage(generateImage.numpy())

    # Saving the Generate Image
    if SAVE_EVERY:
      if iter % SAVE_EVERY == 0:
        new = deprocessImage(generateImage.numpy())
        cv2.imwrite(f'generateImage_{iter+1}.jpg', cv2.cvtColor(new, cv2.COLOR_BGR2RGB))

    
  
  bestImage = deprocessImage(generateImage.numpy())
  cv2.imwrite(f'FINAL_OUTPUT.jpg', cv2.cvtColor(bestImage, cv2.COLOR_BGR2RGB))
  # Saving the numpy Arrays to plot later
  np.save('contentLoss.npy', contentCostLog)
  np.save('styleLoss.npy', styleCostLog)
  np.save('totalCostLoss.npy', totalCostLog)

  return bestImage, (contentCostLog, styleCostLog, totalCostLog)
