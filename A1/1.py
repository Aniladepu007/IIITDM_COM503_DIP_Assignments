import methods as f
n = 5
a =[]
print('enter 25 intergers on each new line :')
for i in range(n):
    temp =[]
    for j in range(n):
        temp.append(int(input()))
    a.append(temp)

Sum = 0
for i in a:
    Sum += sum(i)
print(Sum)

max = -1
for i in a:
    localMax = f.findMax(i)
    if localMax > max:
        max = localMax
print(max)

mean = Sum/(n*n)
print(mean)

med = []
for i in a:
    for j in i:
        med.append(j)
med.sort()
print(med,"\n\n",med[len(med)//2])

dict = {}
for i in range(1,11):
    dict.__setitem__(i,0)

for i in a:
    for j in i:
        dict[j] += 1
val = list(dict.values())
maxVal = sorted(val)[len(val)-1]

mode = []
for i in range(0,len(val)):
    if val[i] == maxVal:
        mode.append(i+1)
print(mode)
print(dict)
std=0
for i in med:
    std += (i - mean)**2
std /= len(med)
print(std**0.5)
