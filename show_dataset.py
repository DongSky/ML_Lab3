from matplotlib import pylab as plt

if __name__ == "__main__":
    f = open("dataset.txt","r")
    x1 = []
    x2 = []
    x3 = []
    y1 = []
    y2 = []
    y3 = []
    for i in f.readlines():
        piece = i.split()
        if int(piece[0]) == 1:
            x1.append(float(piece[1]));y1.append(float(piece[2]))
        if int(piece[0]) == 2:
            x2.append(float(piece[1]));y2.append(float(piece[2]))
        if int(piece[0]) == 3:
            x3.append(float(piece[1]));y3.append(float(piece[2]))
    ax = plt.figure().add_subplot(111)
    ax.scatter(x1,y1,c='r')
    ax.scatter(x2,y2,c='g')
    ax.scatter(x3,y3,c='b')
    plt.show()
