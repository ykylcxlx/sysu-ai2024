import numpy as np
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
def tanh(x):
    return np.tanh(x)
class HiddenLayer():
    def __init__(self,input,n_in,n_out,W=None,b=None,activation=tanh):
        #input size:（n_example,n_in）
        self.input = input
        if W is None:
            self.W = np.asarray(np.random.uniform(
                low = -np.sqrt(6/(input.shape[1]+1)),
                high = np.sqrt(6/(input.shape[1]+1)),
                size=(input.shape[1],1)))
        else:
            self.W = W
        if b is None:
            self.b = np.random.randn(1)
        else:
            self.b = b
        self.params = [self.W, self.b]
        self.output = tanh(np.dot(self.input,self.W)+b)
class Softmax:
    def __init__(self,input,n_in,n_out):
        self.input = input
        self.W = np.random.randn(n_in,n_out)
        self.b = np.random.randn(n_out)
        self.output = np.dot(self.input,self.x,n_in,n_out)
        self.y_pred = np.argmax(self.output,axis=1)#取下标
        self.params = [self.W,self.b]



class MLP:
    def __init__(self,input,input_size,n_in,n_hidden,n_out):
        
        self.hiddenLayer = HiddenLayer(
            input=input
            n_in = n_in
            n_out = n_out
        )
        self.Softmax = Softmax(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        

        self.params = [self.hiddenLayer.params]+self.Softmax.params
        """
        numpy.asarray()函数的基本功能是将输入转换为NumPy数组。这个函数非常灵活，可以接受多种类型的输入，包括列表、元组、其他数组等，并尝试将它们转换为NumPy数组。如果输入已经是NumPy数组，
        那么numpy.asarray()将直接返回原数组，而不会进行复制
        """
        
    def model(self,X,w,y):
         input_size = X.shape[1]
         hidden_size = 3
         output_size = 1
         return sigmoid(np.dot(X,w))
    def gd(self,w,b,x,y,lr):#lr为学习率
    #注意二维及以下的向量才能进行点积运算
    #损失函数对w求偏导
        d_w0 = -np.dot((y - sigmoid(np.dot(w.T, x) + b)).reshape(-1, 1).T, x[:, 0]).reshape(1) / len(x)#因为loss函数是有负号的
        #注意求导数是前面的符号
        d_w1 = -np.dot((y-sigmoid(np.dot(w.T,x)+b)).reshape(-1,1).T,x[:,1]).reshape(1)/len(x)
        d_b = -np.mean(y-sigmoid(np.dot(w.T,x)+b))
        #w = np.random.randn(X.shape[1])#从标准正态分布中随机取值，作为初始值
        #注意reshanpe前面括号的位置
        #d_w0 = -np.dot((y - sigmoid(np.dot(w.T, x) + b)).reshape(-1, 1).T, x[:, 0]).reshape(1) / len(x)
        w[0] = w[0]-self.lr*d_w0
        w[1] = w[1]-self.lr*d_w1
        b = b - self.lr * d_b
        return w,b
    def train(self):
        X,y = read_csv("data/data.csv")
        for i in range(X.shape[1]-1):
            X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])#归一化
        losses = []
        w = np.random.randn(X.shape[1])
        for i in range(1000):
            w, b = self.gd(w,b,X,y,self.lr)
            y_predict = self.model(w,b,X)
            ls = loss(y,y_predict)
            #acc = ((y_predict>0.5) ==y).astype(np.int32).sum()/len(y) *100
            acc = np.sum(np.array((y_predict>0.5) ==y) )/len(X)*100
            losses.append(ls)
            print('Epoch:{}/{} Loss:{:.4f} acc:{:.2f}%'.format(i,1000,ls,acc))
        print('the model is y = {:.3f}x1 + {:.3f}x2 + {:.3f}'.format(w[0][0],w[1][0],b[0]))
        plt.plot(losses)
        plt.show()
         

if __name__ == "__main__":
    # mylr = LogisticRegression()
    # mylr.train()
    mymp = MP()
    mymp.train()
