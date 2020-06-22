import numpy as np
import random as r
import cv2
import matplotlib.pyplot as plt

def sp_noise(img, prob):
    output = img
    for i in range(len(img)):
        for j in range(len(img[0])):
            rd = r.random()
            if rd < prob:
                output[i][j] = 0
            elif rd > 1-prob:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output

img = cv2.imread('lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
avg = []

fig = plt.figure()
rd = 0.001
for i in range(6):
    rd += 0.005
    img = cv2.imread('lena.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = fig.add_subplot(2,3,i+1)
    out = sp_noise(img,rd)
    avg.append(out)
    plt.imshow(out,cmap="gray")
    a.set_title(rd)
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

plt.show()

avg = np.mean(avg,axis=0)
plt.imshow(avg,cmap="gray")
plt.show()
