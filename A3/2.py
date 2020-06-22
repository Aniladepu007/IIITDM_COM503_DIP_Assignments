import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def bilinear_interpolation(x,y,x1,y1,x2,y2,q11,q12,q21,q22):
    sum = q11 * (x2 - x) * (y2 - y) + q21 * (x2 - x) * (y - y1) + q12 * (x - x1) * (y2 - y) + q22 * (x - x1) * (y - y1)
    return (sum) // ((x2 - x1) * (y2 - y1))

img = cv2.imread('pisa1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
angle = 4.5
angle = angle*math.pi/180
cosx = math.cos(angle)
sinx = math.sin(angle)
affinity_matrix = np.array([[cosx,-sinx],[sinx,cosx]])
rot_img = np.zeros(gray_img.shape)

for i in range(gray_img.shape[0]):
  for j in range(gray_img.shape[1]):
    x = np.array([[i],[j]])
    loc = np.matmul(affinity_matrix,x)
    x_cord = int(round(loc[0][0]))
    y_cord = int(round(loc[1][0]))
    if(x_cord>=0 and y_cord>=0 and x_cord<gray_img.shape[0] and y_cord<gray_img.shape[1]):
      rot_img[x_cord][y_cord] = gray_img[i][j]

plt.imshow(rot_img,cmap="gray")
plt.show()

for i in range(rot_img.shape[0]-1,-1,-1):
  for j in range(rot_img.shape[1]-1,-1,-1):
    if(rot_img[i][j] == 0):
      x1 = i-1
      y1 = j-1
      x2 = (i+1)
      y2 = (j+1)
      if(x2>=rot_img.shape[0] or y2>=rot_img.shape[1]):
          rot_img[i][j] = 0
      else:
        rot_img[i][j] = bilinear_interpolation(i, j , x1, y1, x2, y2, rot_img[x1][y1], rot_img[x1][y2], rot_img[x2][y1], rot_img[x2][y2])

plt.imshow(rot_img,cmap="gray")
plt.show()
