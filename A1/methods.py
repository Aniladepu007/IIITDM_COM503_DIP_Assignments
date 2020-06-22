import random as r

def seed(m,n,start,end):
    a = []
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(r.randrange(start,end))
        a.append(tmp)
    return a

def findMax(l):
    return sorted(l)[len(l)-1]
