import numpy as np
import matplotlib.image as img
from statistics import mean
import statistics
import matplotlib.pyplot as plt
import random
import cv2

def gray_scale(m):
    gray = np.zeros([len(m),len(m[0]),3])
    for i in range(len(m)):
        for j in range(len(m[0])):
            lst = [float(m[i][j][0]), float(m[i][j][1]), float(m[i][j][2])]
            avg = float(mean(lst))
            gray[i][j][0] = avg
            gray[i][j][1] = avg
            gray[i][j][2] = avg
    return(gray)

def median_filter(I,kernel):
    I_new = I
    k=int(kernel/2)
    count=0
    for i in range(k,len(I)-k):
        for j in range(k,len(I[0])-k):
            x=[]
            for l1 in range(-k,k+1):
                for l2 in range(-k,k+1):
                    p=I[i+l1][j+l2]
                    x.append(p[0])
                    #print(p)
            x.sort()
            count=count+1
            #print(statistics.median(x))
            I_new[i][j]=statistics.median(x)
    print(count)
    return(I_new)

def add_padding_k(gray1,kernel):
    k=int(kernel/2)
    gray2 = np.zeros([len(gray1)+(2*k),len(gray[0])+(2*k),3])
    l=(len(gray2))
    m=(len(gray2[0]))
    print(l,m)
    for i in range(k,l-k):
        for j in range(k,m-k):
            gray2[i][j]=gray1[i-k][j-k]
    return(gray2)

def sp_noise(image,p):
    count=0
    count1=0
    output = image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.randint(0,p)
            if rdn == 0:
                output[i][j] = 0
                count=count+1
            elif rdn==p:
                output[i][j] = 1
                count1=count1+1
            else:
                output[i][j] = image[i][j]
    print(count)
    print(count1)
    return(output)

def average_filter_k(I,kernel):
    I_new = np.zeros([len(I),len(I[0]),3])
    k=int(kernel/2)
    for i in range(k,len(I)-k):
        for j in range(k,len(I[0])-k):
            for l1 in range(-k,k+1):
                for l2 in range(-k,k+1):
                    #if(i==1 and j==1):
                        #print(l1,l2)
                    I_new[i][j] = I_new[i][j]+I[i+l1][j+l2]
            I_new[i][j]=I_new[i][j]/(kernel*kernel)
    return(I_new)

Lena_image = img.imread("lena.png")
Lena_gray = gray_scale(Lena_image)
plt.imshow(Lena_gray)

gray=Lena_gray.copy()
n1 = sp_noise(gray,int(255*(1)/10))
n1_padded = add_padding_k(n1,3)
n1_new=average_filter_k(n1_padded,3)
plt.imshow(n1_new)

n1_padded_k=add_padding_k(n1,5)
n1_temp5=average_filter_k(n1_padded_k,5)
plt.imshow(n1_temp5)

n1_padded_k7=add_padding_k(n1,7)
n1_temp7=average_filter_k(n1_padded_k7,7)
plt.imshow(n1_temp7)

n1_median=median_filter(n1,3)
plt.imshow(n1_median)

n1_median=median_filter(n1,5)
plt.imshow(n1_median)

gray=Lena_gray.copy()
n2 = sp_noise(gray,int(255*(2)/10))
n2_padded = add_padding_k(n2,3)
n2_new=average_filter_k(n2_padded,3)
plt.imshow(n2_new)

n2_padded_k=add_padding_k(n2,5)
n2_temp5=average_filter_k(n2_padded_k,5)
plt.imshow(n2_temp5)

n2_padded_k7=add_padding_k(n2,7)
n2_temp7=average_filter_k(n2_padded_k7,7)
plt.imshow(n2_temp7)

n2_median=median_filter(n2,3)
plt.imshow(n2_median)

gray=Lena_gray.copy()
n3 = sp_noise(gray,int(255*(3)/10))
n3_padded = add_padding_k(n3,3)
n3_new=average_filter_k(n3_padded,3)
plt.imshow(n3_new)

n3_median=median_filter(n3,3)
plt.imshow(n3_median)

n3_padded_k=add_padding_k(n3,5)
n3_temp5=average_filter_k(n3_padded_k,5)
plt.imshow(n3_temp5)

n3_padded_k7=add_padding_k(n3,7)
n3_temp7=average_filter_k(n3_padded_k7,7)
plt.imshow(n3_temp7)

gray=Lena_gray.copy()
n4 = sp_noise(gray,int(255*(4)/10))
n4_padded = add_padding_k(n4,3)
n4_new=average_filter_k(n4_padded,3)
plt.imshow(n4_new)

n4_median=median_filter(n4,3)
plt.imshow(n4_median)

n4_padded_k=add_padding_k(n4,5)
n4_temp5=average_filter_k(n4_padded_k,5)
plt.imshow(n4_temp5)

n4_padded_k7=add_padding_k(n4,7)
n4_temp7=average_filter_k(n4_padded_k7,7)
plt.imshow(n4_temp7)

gray=Lena_gray.copy()
n5 = sp_noise(gray,int(255*(5)/10))
n5_padded = add_padding_k(n5,3)
n5_new=average_filter_k(n5_padded,3)
plt.imshow(n5_new)

n5_median=median_filter(n5,3)
plt.imshow(n5_median)

n5_padded_k=add_padding_k(n5,5)
n5_temp5=average_filter_k(n5_padded_k,5)
plt.imshow(n5_temp5)

n5_padded_k7=add_padding_k(n5,7)
n5_temp7=average_filter_k(n5_padded_k7,7)
plt.imshow(n5_temp7)

gray=Lena_gray.copy()
n6 = sp_noise(gray,int(255*(6)/10))
n6_padded = add_padding_k(n6,3)
n6_new=average_filter_k(n6_padded,3)
plt.imshow(n6_new)

n6_median=median_filter(n6,3)
plt.imshow(n6_median)

n6_padded_k=add_padding_k(n6,5)
n6_temp5=average_filter_k(n6_padded_k,5)
plt.imshow(n6_temp5)



n6_padded_k7=add_padding_k(n6,7)
n6_temp7=average_filter_k(n6_padded_k7,7)
plt.imshow(n6_temp7)

gray=Lena_gray.copy()
n7 = sp_noise(gray,int(255*(7)/10))
n7_padded = add_padding_k(n7,3)
n7_new=average_filter_k(n7_padded,3)
plt.imshow(n7_new)

n7_median=median_filter(n7,3)
plt.imshow(n7_median)

n7_padded_k=add_padding_k(n7,5)
n7_temp5=average_filter_k(n7_padded_k,5)
plt.imshow(n7_temp5)

n7_padded_k7=add_padding_k(n7,7)
n7_temp7=average_filter_k(n7_padded_k7,7)
plt.imshow(n7_temp7)

gray=Lena_gray.copy()
n8 = sp_noise(gray,int(255*(8)/10))
n8_padded = add_padding_k(n8,3)
n8_new=average_filter_k(n8_padded,3)
plt.imshow(n8_new)

n8_median=median_filter(n8,3)
plt.imshow(n8_median)

n8_padded_k=add_padding_k(n8,5)
n8_temp5=average_filter_k(n8_padded_k,5)
plt.imshow(n8_temp5)

n8_padded_k7=add_padding_k(n8,7)
n8_temp7=average_filter_k(n8_padded_k7,7)
plt.imshow(n8_temp7)

gray=Lena_gray.copy()
n9 = sp_noise(gray,int(255*(9)/10))
n9_padded = add_padding_k(n9,3)
n9_new=average_filter_k(n9_padded,3)
plt.imshow(n9_new)

n9_median=median_filter(n9,3)
plt.imshow(n9_median)

n9_padded_k=add_padding_k(n9,5)
n9_temp5=average_filter_k(n9_padded_k,5)
plt.imshow(n9_temp5)

n9_padded_k7=add_padding_k(n9,7)
n9_temp7=average_filter_k(n9_padded_k7,7)
plt.imshow(n9_temp7)

gray=Lena_gray.copy()
n10 = sp_noise(gray,int(255*(10)/10))
n10_padded = add_padding_k(n10,3)
n10_new=average_filter_k(n10_padded,3)
plt.imshow(n10_new)

n10_median=median_filter(n10,3)
plt.imshow(n10_median)

n10_padded_k=add_padding_k(n10,5)
n10_temp5=average_filter_k(n10_padded_k,5)
plt.imshow(n10_temp5)

n10_padded_k7=add_padding_k(n10,7)
n10_temp7=average_filter_k(n10_padded_k7,7)
plt.imshow(n10_temp7)

import numpy as np
import cv2

img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1]

for i in np.arange(3, height-3):
    for j in np.arange(3, width-3):
        neighbors = []
        for k in np.arange(-3, 4):
            for l in np.arange(-3, 4):
                a = img.item(i+k, j+l)
                neighbors.append(a)
        neighbors.sort()
        median = neighbors[24]
        b = median
        img_out.itemset((i,j), b)

cv2.imwrite('filter_median.jpg', img_out)

cv2.imshow('image',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
