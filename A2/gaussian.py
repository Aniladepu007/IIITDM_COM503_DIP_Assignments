import numpy as np
import random as r
import cv2
import matplotlib.pyplot as plt

def gauss(image, sigma):
    output = [np.zeros(image.shape,np.uint8)]
    row,col= image.shape
    gauss = np.random.randn(row, col) * sigma

    mean =0
    plt.imshow(gauss,cmap="gray")
    plt.show()
    count, bins, ignored = plt.hist(gauss, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    plt.show()

    output = image + gauss
    return output

img = cv2.imread('lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
avg = []

fig = plt.figure()
rd = 0
for i in range(6):
    rd += 5
    img = cv2.imread('lena.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = fig.add_subplot(2,3,i+1)
    out = gauss(img,rd)
    avg.append(out)
    plt.imshow(out,cmap="gray")
    a.set_title(rd)
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

plt.show()

avg = np.mean(avg,axis=0)
plt.imshow(avg,cmap="gray")
plt.show()
