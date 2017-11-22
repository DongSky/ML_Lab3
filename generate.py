#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    num = 100
    mu1 = [3, 7]
    mu2 = [2, 2]
    mu3 = [8, 5]
    cov = [[2,0],[0,2]]
    f = open("dataset.txt","w")
    x, y = np.random.multivariate_normal(mu1, cov, num ).T
    for i in range(100):
        f.write(" ".join(["1",str(x[i]),str(y[i]),"\n"]))
    #ax = plt.figure().add_subplot(111)
    #ax.scatter(x,y,c='r')
    x, y = np.random.multivariate_normal(mu2, cov, num ).T
    for i in range(100):
        f.write(" ".join(["2",str(x[i]),str(y[i]),"\n"]))
    #ax.scatter(x,y,c='g')
    x, y = np.random.multivariate_normal(mu3, cov, num ).T
    for i in range(100):
        f.write(" ".join(["3",str(x[i]),str(y[i]),"\n"]))
    #ax.scatter(x,y,c='b')
    #plt.show()
    f.close()
