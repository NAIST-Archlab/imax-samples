import cv2
import numpy as np
import matplotlib.pyplot as plt

width = 640
height = 720

res = np.loadtxt('clustering_result_test.csv')
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
myImg = np.float32(myImg)

cv2.imwrite('seg.jpg', myImg)
cv2.imshow("img" , myImg)
