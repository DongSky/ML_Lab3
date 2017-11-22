from matplotlib import pylab as plt
import numpy as np


def read_file(filename):
    f = open(filename, 'r')
    d = f.readlines()
    f.close()
    return d


def gaussian(x_i, mu_i, sigma_i):
    norm_factor = 1.0 / np.linalg.det(sigma_i)
    sgm = norm_factor * np.exp(-0.5 * np.transpose(x_i - mu_i).dot(np.linalg.inv(sigma_i)).dot(x_i - mu_i))
    return sgm


class GMM(object):
    def __init__(self):
        data = read_file('seeds_dataset.txt')
        self.n = len(data) 
        self.dim = 7 
        self.x = np.zeros((self.n, self.dim), dtype='float64') 
        self.y = np.zeros((self.n, 1), dtype='float64') 

        for i in range(len(data)):
            data_ = data[i].split()
            self.y[i] = int(data_[7])
            self.x[i][0] = float(data_[0])
            self.x[i][1] = float(data_[1])
            self.x[i][2] = float(data_[2])
            self.x[i][3] = float(data_[3])
            self.x[i][4] = float(data_[4])
            self.x[i][5] = float(data_[5])
            self.x[i][6] = float(data_[6])

        self.k = 3 
        self.gama = np.zeros((self.n, self.k), dtype='float64')
        self.mu = np.zeros((self.k, self.dim), dtype='float64')  
        self.mu[0] = self.x[30]
        self.mu[1] = self.x[120]
        self.mu[2] = self.x[200]

        self.sigma = np.array([np.identity(self.dim) for i in range(0, self.k)], dtype='float64')  
        self.alpha = np.array([1.0 / self.k for i in range(0, self.k)], dtype='float64')  
        self.epsilon = 0.0001

        self.em()
        print(self.mu)
        print(self.sigma)
        self.gmm()

    def new_mu(self, k):
        mu_k = np.zeros((1, self.dim), dtype='float64')
        for i in range(self.n):
            mu_k += self.gama[i][k] * self.x[i]
        return mu_k / np.sum(self.gama[:, k])

    def new_sigma(self, k):
        sigma_k = np.zeros((self.dim, self.dim), dtype='float64')
        for i in range(self.n):
            sigma_k += self.gama[i][k] * np.transpose(np.mat(self.x[i] - self.mu[k])).\
                dot(np.mat(self.x[i] - self.mu[k]))
        return sigma_k / np.sum(self.gama[:, k])

    def new_alpha(self, k):
        return np.sum(self.gama[:, k]) / self.n

    def Q(self):
        q = 1
        for i in range(self.n):
            p = 0
            for j in range(self.k):
                p += self.alpha[j] * gaussian(self.x[i], self.mu[j], self.sigma[j])
            q *= p
        return q

    def em(self):
        # p_new = self.epsilon * 2
        # p_old = 0
        # while abs(p_new - p_old) > self.epsilon:
            # p_old = p_new
        for k in range(20):
            for i in range(self.n):
                sum_gama_k = 0
                for j in range(self.k):
                    sum_gama_k += self.alpha[j] * gaussian(self.x[i], self.mu[j], self.sigma[j])

                for j in range(self.k):
                    self.gama[i][j] = (self.alpha[j] * gaussian(self.x[i], self.mu[j], self.sigma[j])) / sum_gama_k
            for i in range(self.k):
                self.alpha[i] = self.new_alpha(i)
                self.sigma[i] = self.new_sigma(i)
                self.mu[i] = self.new_mu(i)
            # p_new = self.Q()

    def gmm(self):
        count1 = 0
        count2 = 0
        count3 = 0
        for i in range(self.n):
            p1 = self.alpha[0] * gaussian(self.x[i], self.mu[0], self.sigma[0])
            p2 = self.alpha[1] * gaussian(self.x[i], self.mu[1], self.sigma[1])
            p3 = self.alpha[2] * gaussian(self.x[i], self.mu[2], self.sigma[2])
            if max(p1, p2, p3) == p1:
                count1 += 1
            elif max(p1, p2, p3) == p2:
                count2 += 1
            else:
                count3 += 1

        x_1 = np.zeros((count1, self.dim), dtype='float64')
        x_2 = np.zeros((count2, self.dim), dtype='float64')
        x_3 = np.zeros((count3, self.dim), dtype='float64')

        count1 = 0
        count2 = 0
        count3 = 0
        correct = 0
        for i in range(self.n):
            p1 = self.alpha[0] * gaussian(self.x[i], self.mu[0], self.sigma[0])
            p2 = self.alpha[1] * gaussian(self.x[i], self.mu[1], self.sigma[1])
            p3 = self.alpha[2] * gaussian(self.x[i], self.mu[2], self.sigma[2])
            if max(p1, p2, p3) == p1:
                x_1[count1] = self.x[i]
                count1 += 1
                if self.y[i] == 1: correct += 1
            elif max(p1, p2, p3) == p2:
                x_2[count2] = self.x[i]
                count2 += 1
                if self.y[i] == 2: correct += 1
            else:
                x_3[count3] = self.x[i]
                count3 += 1
                if self.y[i] == 3: correct += 1
        print(count1,count2,count3)
        print(correct / self.n)

if __name__ == '__main__':
    a = GMM()