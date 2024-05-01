import cv2
import numpy as np
import matplotlib.pyplot as plt

width = 590
height = 427

res = np.loadtxt('clustering_result.csv')
myImg = None
for re in res:
    r = (re % 5) * 51
    g = (re % 4) * 60
    b = (re % 3) * 85
    if myImg is None:
        myImg = [r , g , b]
    else:
        myImg = np.vstack( [myImg , [r,g,b]] )

myImg = myImg.reshape( height , width , 3 )
cv2.imshow(myImg)

