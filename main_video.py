from __helpers__ import *


VIDEO_PATH       = "/content/drive/My Drive/Colab Notebooks/Neural Style Transfer/vids/surf.mp4" #@param {type:"string"}
STYLE_IMAGE_PATH = "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ8LOCDfTvcJ_V4fBdtL3R_oQn7D9P96PPzJFCksdWeKHHhyfUZ" #@param {type:"string"}

FPS_QUALITY      = "high" 
'''
high   > original fps > slowest
medium > 
low    > bad fps      > fastest
''' 

STYLE_QUALITY    = 100 # Any integer in range (100, 1000)
'''
The more the STYLE_QUALITY, the more better every frame is stylized.
BUT, will take more time.
'''

# Final Output video name
OUTPUT_FILE_NAME = 'trying_aivayy'




generated_frames = []
vid_frames = extract_frames_out_of_the_video(VIDEO_PATH)

for frame_number in tqdm(range(0, len(vid_frames), skip_frame_every(FPS_QUALITY))):

  content = vid_frames[frame_number]

  bestImage, bestLoss, output_dirName = runStyleTransfer(content,
                                                        STYLE_IMAGE_PATH,
                                                        iterations=style_quality,
                                                        contentWeight = 1,
                                                        styleWeight= 0.8)
  generated_frames.append(bestImage)
  height, width, channels = bestImage.shape



PATH = os.path.join(os.curdir, 'outputs')
if not os.path.isdir(PATH):
  os.mkdir(PATH)
os.chdir(PATH)

print('Developing the Video...')
out = cv2.VideoWriter(OUTPUT_FILE_NAME + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width,height))
for i in (range(len(generated_frames))):
    out.write(generated_frames[i])
out.release()

print('Video Converted and Saved Succesfully!')