import matplotlib.pyplot as plt

from __helpers__ import *

###################################################################
# Few templates
wave        = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ8LOCDfTvcJ_V4fBdtL3R_oQn7D9P96PPzJFCksdWeKHHhyfUZ'
seated_nude = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ0ip7KMW5XB_qhU3cwBDDd1fjlogHfgOxw9gnVq2CqZdLwHgY3'
shinchan    = 'https://pbs.twimg.com/profile_images/452516792426975232/rOQPTVq4_400x400.png'
###################################################################

STYLE_QUALITY = 1
DIR_NAME      = 'seated_nude' # Set this to None if outputs are not required to be saved locally.

contentImagePath = shinchan
styleImagePath   = seated_nude

generatedImage = runStyleTransfer(contentImagePath,
                                  styleImagePath,
                                  iterations=STYLE_QUALITY,
                                  SAVE_EVERY = 0,
                                  contentWeight = 1,
                                  styleWeight= 0.8,
                                  output_dirName = DIR_NAME)


print('Plotting...')

fig = plt.figure(constrained_layout=False, figsize=(10, 10))
gs1 = fig8.add_gridspec(nrows=2, ncols=3, left=0.005, right=0.48, wspace=0.05, hspace=0.01)

fig.add_subplot(gs1[:-1, :])
plt.imshow(bestImage)
plt.axis('off')
plt.title('Generated Image')

fig.add_subplot(gs1[-1, :-1])
plt.imshow(inputImageAndPreprocess(contentImagePath)[0])
plt.axis('off')
plt.title('Content Image')

fig.add_subplot(gs1[-1, -1])
plt.imshow(inputImageAndPreprocess(styleImagePath)[0])
plt.axis('off')
plt.title('Style Image')



plt.figure(figsize=(23, 8))
plt.plot(totalCostLog,   linewidth=3, label='total loss')
plt.plot(styleCostLog,   linewidth=1, label='style loss')
plt.plot(contentCostLog, linewidth=2, label='content loss')
# plt.plot(learning_curve_tv, linewidth=2, label='total variation loss')
plt.title("Learning curve")
plt.ylabel("error")
plt.xlabel("epoch")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()

print('All done yay!')