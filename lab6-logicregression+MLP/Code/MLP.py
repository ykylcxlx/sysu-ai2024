import numpy as np
import matplotlib.pyplot as plt

rate=0.05 #学习率
# sample_num=400 #样本数据量
def read_csv(path):
    data = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=1,usecols=range(3))
    X = data[:,0:-1]
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])#归一化
    #X = np.insert(X, len(X[0]), 1, axis=1)#最后一列全插入1

    '''np.insert()函数的参数如下：
    X：要插入值的输入数组。
    0：要插入的位置索引。在这种情况下，0表示要在数组的开头插入值。
    1：要插入的值。在这种情况下，插入的值是1。
    axis=1：指定要插入值的轴。axis=1表示在数组的列维度上插入。
    '''
    class_label = data[:,-1]
    class_label = np.expand_dims(class_label, axis=1)
    return X,class_label
class my_mlp:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.normal(size=(input_size, hidden_size))#输入层到隐藏层
        self.w2 = np.random.normal(size=(hidden_size,output_size))#隐藏层到输出层
        self.b1 = np.random.normal(size=(hidden_size))
        self.b2 = np.random.normal(size=(output_size))
        self.h_out = np.zeros(1)
        self.out = np.zeros(1)

    @staticmethod
    def sigmoid(x):
        '''sigmoid函数作为激活函数'''
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def d_sigmoid(x):
        '''相对误差对输出和隐含层求导'''
        return x * (1 - x)
    def forward(self,input):
        # print(np.shape(input),np.shape(self.w1),np.shape(self.w2))
        # print(np.shape(np.dot(input,self.w1)))
        self.h_out = my_mlp.sigmoid(np.dot(input, self.w1)+self.b1)
        self.out = my_mlp.sigmoid(np.dot(self.h_out, self.w2)+self.b2)
        return self.out
    def test(self,input,y):
        cnt = 0
        self.out = self.forward(input)
        res = abs(self.out-y)
        for i in range(len(y)):
            if res[i]<0.5:
                cnt+=1
        # for i in range(len(y)):
        #     if(self.out[i]>=0.5):
        #         if y[i]==1:
        #             cnt+=1
        #     if(self.out[i]<0.5):
        #         if y[i]==0:
        #             cnt+=1
        return cnt/len(y)
    def backpropagation(self,input,output,lr=rate):
        sample_num = input.shape[0]
        self.forward(input)
        # print(np.shape(output))
        # print(np.shape(self.out))
        #求对层输出的梯度
        L2_delta=(output-self.out)
        #print(np.shape(L2_delta))
        L1_delta = L2_delta.dot(self.w2.T) * my_mlp.d_sigmoid(self.h_out)
        #求对参数的梯度
        d_w2 = rate * self.h_out.T.dot(L2_delta)
        #print(d_w2.shape)
        d_w1 = rate * input.T.dot(L1_delta)
        self.w2 += d_w2
        self.w1 += d_w1
        d_b2 = np.ones((1,sample_num)).dot(L2_delta)
        d_b1 = np.ones((1,sample_num)).dot(L1_delta)
        self.b2 += rate*d_b2.reshape(d_b2.shape[0]*d_b2.shape[1],)
        self.b1 += rate*d_b1.reshape(d_b1.shape[0]*d_b1.shape[1],)
def draw_decision_boundaries(X,y,mlp):
    # 确定绘图范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    # print(np.shape(xx.ravel()))
    # print(np.shape(yy))
    # 对网格点进行预测
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = np.where(Z > 0.5, 1, 0)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of MLP')
    plt.show()

if __name__ == '__main__':
    mlp=my_mlp(2,2,1)
    # x_data x1,x2
    x_data = read_csv("data/data.csv")[0]
    y_data = read_csv("data/data.csv")[1]
    data_size = x_data.shape[0]
    batch_size = 50
    losses = []
    acc_list = []
    for i in range(5000):
        indices = np.random.choice(data_size, batch_size)
        x_batch = x_data[indices]
        y_batch = y_data[indices]
        mlp.backpropagation(x_batch,y_batch) #反向传播
        out=mlp.forward(x_data)
        loss = -np.mean(y_data*np.log(out)+(1-y_data)*np.log(1-out))
        # if i % 500 == 0:
        #     plt.scatter(i, np.mean(np.abs(y_data - out)))
        losses.append(loss)
        #acc_list.append(mlp.test(x_data,y_data))
            #print('当前误差:',np.mean(np.abs(y_data - out)))
    plt.plot(losses)
    #plt.plot(acc_list)
    plt.title('Loss Curve')
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.show()
    draw_decision_boundaries(x_data,y_data,mlp)
    print('输入层到隐含层权值:\n',mlp.w1)
    print('输入层到隐含层偏置：\n',mlp.b1)
    print('隐含层到输出层权值：\n',mlp.w2)
    print('隐含层到输出层偏置：\n',mlp.b2)

    print('输出结果:\n',out)
    
    print('交叉熵损失函数的值:',losses[-1])
    print("准确率:",mlp.test(x_data,y_data))
    #print('忽略误差近似输出:')
    # for i in out:
    #     print(0 if i<=0.5 else 1)
