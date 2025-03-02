# %%
import numpy as np
import matplotlib.pyplot as plt
def read_csv(filename):
    arr = np.loadtxt(filename,
                 delimiter=",", skiprows=0, dtype=str)
    features = arr[0]
    arr = arr[1:-1]
    return arr,features
data,features = read_csv("DT_data.csv")
print(features)

label_num = 2
X = np.delete(data,obj=label_num,axis=1)
y = np.where(data[:, label_num].astype(int) <= 30000, 1, 0)  # 信誉度分类
features = np.delete(features,label_num)

# %%

# 将文本特征转换为数值特征
X_encoded = np.zeros(X.shape)
for i in range(X.shape[1]):
    if i != 2:  #年龄这一列不需要进行变化
        unique_labels = np.unique(X[:, i])#去重
        label_mapping = {label: j for j, label in enumerate(unique_labels)}#获取字符串和数字组成的键值对
        X_encoded[:, i] = np.array([label_mapping[label] for label in X[:, i]])

    else:
        X_encoded[:,i] = X[:,i]
#print(y)



# %%
class Node:
    def __init__(self, feature=None, threshold=None, value=None,num=0):
        self.feature = feature  # 特征索引
        self.threshold = threshold # 分割阈值
        self.left = None
        self.right = None
        self.value = value # 叶子节点的类别
        self.num = num # 叶子节点的样本数量
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
    def gini_index(self,labels):
        #np.unique函数
        _,counts = np.unique(labels,return_counts=True)
        probs = counts / len(labels)
        gini = 1-np.sum(probs ** 2)
        return gini
    def split_dataset(self,X,y,feature,threshold):
        #获取是否小于threshold的掩码
        left_mask = X[:,feature] <= threshold
        right_mask = X[:,feature] > threshold
        #print('y[right_mask]',y[right_mask])
        return X[left_mask],y[left_mask], X[right_mask], y[right_mask]
    def get_best_split(self,X,y):
        best_feature,best_threshold, gini =None, None, np.inf
        y_left = y_right = None
        #下面进行遍历，以基尼系数最小为标准，获取最佳分裂点
        for i in range(X.shape[1]):
            thresholds = np.unique(X[:,i])
            #print(thresholds)
            for threshold in thresholds:
                X_left,y_left,X_right,y_right = self.split_dataset(X,y,i,threshold)
                tmp_gini = (len(y_left)*self.gini_index(y_left) + len(y_right)*self.gini_index(y_right))/len(y)
                if tmp_gini < gini:
                    gini = tmp_gini
                    best_threshold = threshold
                    best_feature = i
        #print("best_feature ={} ,best_threshold={} ".format(best_feature,best_threshold))
        X_left,y_left,X_right,y_right = self.split_dataset(X,y,best_feature,best_threshold)
        #print("y_left and right:",y_left,y_right)
        if len(y_left)==0 or len(y_right)==0:
            # print("best_feature ={} ,best_threshold={} ".format(best_feature,best_threshold))
            # print(best_feature,best_threshold)
            # X_left,y_left,X_right,y_right = self.split_dataset(X,y,best_feature,best_threshold)
            # print('y_left',y_left,'y_right',y_right)
            return None,None,1
        return best_feature,best_threshold,gini
    def build_tree(self,X,y,depth=0):
        #y中出现最多的数字
        label = np.argmax(np.bincount(y))

        if len(np.unique(y))==1:
        #所有子节点均为同一类
            #print(y[0])
            return Node(value=y[0], num=len(y))
        if depth>self.max_depth:
            return Node(value=label, num=len(y))
        best_feature,best_threshold,gini = self.get_best_split(X,y)
        if best_feature==None:
            return Node(value=label, num=len(y))
        #print("best_feature ={} ,best_threshold={} , depth={} ".format(features[best_feature],best_threshold,depth))

        node = Node(feature=best_feature,threshold=best_threshold,num= len(y),value=label)
        #切分数据
        X_left,y_left,X_right,y_right = self.split_dataset(X,y,best_feature,best_threshold)
        #递归
        node.left = self.build_tree(X_left,y_left,depth+1)
        node.right = self.build_tree(X_right,y_right,depth+1)
        return node
    #再封装一层
    def fit(self,X,y):
        self.root = self.build_tree(X,y)
    #剪枝
    def prune_tree(self,node,X,y):
        
        if node.left == None and node.right == None:
            return
        if node.left != None:
           self.prune_tree(node.left,X,y)
        if node.right != None:
           self.prune_tree(node.right,X,y)# 这里没有return

        acc_before = self.cal_acc(X,y)
        temp_left = node.left
        temp_right = node.right
        if(temp_left.num>temp_right.num):
                node.value = temp_left.value
        else:
                node.value = temp_right.value

        node.left = None
        node.right = None   
        #print(temp_left==None)
        acc_after = self.cal_acc(X,y)

        if (acc_after<acc_before):
            node.left = temp_left
            node.right = temp_right

        # else:


    #预测新样本label
    def predict(self,X):
        
        y = np.zeros(X.shape[0])
        for index,sample in enumerate(X):
            node = self.root
            while node.left !=None and node.right != None:
                feature = node.feature
                #print("node.threshold ",node.threshold)
                #print(sample[feature])
                if sample[feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            assert node.value!=None
            y[index] = node.value
        return y
    #计算准确率
    def cal_acc(self,X,y):
        cnt = 0
        pred = self.predict(X)
        #print(y)
        acc = np.sum(pred==y)/len(y)
        return acc

# %%
# 设置随机数种子,确保每次划分结果一致
np.random.seed(42)
# 计算测试集的样本数
test_size = int(X.shape[0] * 0.2)
K = 5 #5格交叉验证
fold_size = X.shape[0] // K
acc_list = []
# 重复 5 次划分过程
for i in range(K):
    val_start = i*fold_size
    val_end = (i+1)*fold_size
    np.random.seed(i)
    
    indices = np.random.permutation(data.shape[0])
    # 按 80:20 的比例划分训练集和测试集
    # 计算测试集的样本数
    test_size = int(data.shape[0] * 0.2)
    train_indices = np.concatenate((indices[:val_start], indices[val_end:])) #数组拼接
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X_encoded[train_indices, :]
    y_train = y[train_indices]
    X_val = X_encoded[val_start:val_end,:]
    y_val = y[val_start:val_end]
    X_test = X_encoded[test_indices, :]
    y_test = y[test_indices]
    
    # print(f"\n划分结果 {i+1}:")
    # print("训练集:")
    # print(X_train)
    # print(y_train)
    # print("测试集:")
    # print(X_test)
    # print(y_test)
    print('第{}次交叉验证'.format(i+1))
    tree = DecisionTree(max_depth=10)
    tree.fit(X_train,y_train)
    print('accuracy before prune',tree.cal_acc(X_test,y_test))
    tree.prune_tree(tree.root,X_val,y_val)
    print('accuracy after prune ',tree.cal_acc(X_test,y_test))
    acc = tree.cal_acc(X_test,y_test)
    acc_list.append(acc)
print(acc_list)


# %%
def traverse(node,depth=0):
    if node==None:
        return 
    if node.left is None and node.right is None:
        print(node.value)
        return
    print("father_Node",features[node.feature],node.threshold,depth)
    traverse(node.left,depth+1)
    traverse(node.right,depth+1)


# %%