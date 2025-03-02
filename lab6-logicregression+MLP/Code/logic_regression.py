import numpy as np
import matplotlib.pyplot as plt

def read_csv(path):
    data = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=1,usecols=range(3))
    X = data[:,0:-1]
    X = np.insert(X, len(X[0]), 1, axis=1)#最后一列全插入1
    '''np.insert()函数的参数如下：
    X：要插入值的输入数组。
    0：要插入的位置索引。在这种情况下，0表示要在数组的开头插入值。
    1：要插入的值。在这种情况下，插入的值是1。
    axis=1：指定要插入值的轴。axis=1表示在数组的列维度上插入。
    '''
    class_label = data[:,-1]
    return X,class_label
def sigmoid(x):
    return  1. / (1. + np.exp(-x))#np.exp函数
def loss(y_hat,y):
            return np.sum(-(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))/y.shape[0]
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
class LogisticRegression:
    def __init__(self,max_iter=3000,lr=0.01):
         self.max_iter = max_iter
         self.lr = lr

    def model(self,X,w,y):
        return sigmoid(np.dot(X,w))
    
    def train(self):
         losses = []
         cnt = 0
         X, y = read_csv("data/data.csv")
         X[:, 0] = (X[:, 0] - np.mean(X[:, 0]))/ np.std(X[:, 0]) # 第0列均值方差归一化
         X[:, 1] = (X[:, 1] - np.mean(X[:, 1]))/ np.std(X[:, 1])  # 第1列均值方差归一化
         w = np.random.randn(X.shape[1])#从标准正态分布中随机取值，作为初始值
         y_hat = self.model(X,w,y)
         for i in range(self.max_iter):
              #grad = X.T*np.sum((y_hat - y))/len(y)
              grad = X[:].T.dot(y_hat - y)/len(y)
            #   w[2] = -np.mean(y-y_hat+w[2])
              #grad = np.concatenate((grad,w[2]), axis=0)
              np.append(grad,w[2])
              w = w - self.lr*grad # 更新权重
              y_hat = self.model(X,w,y) #计算出y_hat
              losses.append(loss(y_hat,y)) #loss append到losses中
              if (abs(loss(y_hat,y))<1e-6):
                   break;
         
         plt.plot(losses)
         
         plt.show()
         self.plotBestFit(w)
         res = abs(y_hat-y)
         for i in range(len(y)):
            if res[i]<0.5:
                cnt+=1
         print("acc:",cnt/len(y))

    def plotBestFit(self,w):
         X,y = read_csv("data/data.csv")
         xxcord1 = []
         xycord1 = []
         xxcord2 = []
         xycord2 = []
        #  X[:, 0] = (X[:, 0] - np.mean(X[:, 0]))/ np.std(X[:, 0]) # 第0列均值方差归一化
        #  X[:, 1] = (X[:, 1] - np.mean(X[:, 1]))/ np.std(X[:, 1])  # 第1列均值方差归一化
         for i in range(len(X)):
              if y[i] == 1:
                  xxcord1.append(X[i,0])
                  xycord1.append(X[i,1])
              else:
                  xxcord2.append(X[i,0])
                  xycord2.append(X[i,1])
         x = np.arange(np.min(X[:,0]),np.max(X[:,0]),1)
         
         #y = -(w[2]+w[0]*x)/w[1]

         y = -(w[2]+w[0]*(x-np.mean(X[:,0]))/np.std(X[:,0]))/w[1]*np.std(X[:,1])+np.mean(X[:,1])
         plt.plot(x,y)
         plt.scatter(xxcord1,xycord1,c='red',s=30)
         plt.xlabel("Age")
         plt.scatter(xxcord2,xycord2,c="g",s=30)
         plt.ylabel("EstimatedSalary")
         plt.show()


if __name__ == "__main__":
    mylr = LogisticRegression()
    mylr.train()