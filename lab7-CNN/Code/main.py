# %%
import os
import time  # 导入time模块
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
    

# %%
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")
print(device)

# %%
#### PATH
path = '/home/yunhao/CNN/cnn图片'
train_dir = os.path.join(path,'train')
test_dir = os.path.join(path,'test')
#存在以‘’/’’开始的参数，从最后一个以”/”开头的参数开始拼接，之前的参数全部丢弃。
#同时存在以‘’./’与‘’/’’开始的参数，以‘’/’为主，从最后一个以”/”开头的参数开始拼接，之前的参数全部丢弃。
print(test_dir)

# %%
# Hyper Parameters
EPOCH = 15               # train the training data n times
BATCH_SIZE = 8
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

# %%
'''获取数据
'''
def generate(dir,label):
	files = os.listdir(dir)
	#print(dir)
	files.sort()
	print('****************')
	print('input :',dir)
	print ('start...')
	listText = open('alltest.txt','a')
	for file in files:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		#print(dir)
		name = dir+'/'+file + '!' + str(int(label)) +'\n'
		name=name.replace('\\','/')
		listText.write(name)
	listText.close()
	print('down!')
	print('****************')
 
 
outer_path = ('/home/yunhao/CNN/cnn图片/test')   #图片的目录
generate(outer_path, 0)

# %%
f = os.listdir('/home/yunhao/CNN/cnn图片/test')
f = f
for file in f:
    filetype = os.path.split(file)
    name = '/home/yunhao/CNN/cnn图片/test'+'/'+ file + '!' + str(1) +'\n'

# %%
'''
数据增强与图象预处理
'''
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),#以给定概率水平对称
    transforms.ToTensor(),#array类型为uint8  经过ToTensor() 后数值由 [0,255] 变为 [0,1]，通过将每个数据除以255进行归一化。
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #别人的解答：数据如果分布在(0,1)之间，可能实际的bias，就是神经网络的输入b会比较大，而模型初始化时b=0的，这样会导致神经网络收敛比较慢，经过Normalize后，可以加快模型的收敛速度。
])
"""构建自己的Dataset"""
'''train_datset'''
train_data = datasets.ImageFolder(train_dir,transform = transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
'''DataLoader 用于将数据集加载为小批量数据'''

test_list = [os.path.join(test_dir,file) for file in os.listdir(test_dir) if file.endswith('.jpg')]

'''test_dataset'''
class TestDataset(Dataset):
    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform
    def __len__(self):
        return len(self.file_list)   # 返回列表元素的数目
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        #打开图片
        img = Image.open(img_name).convert("RGB")
        if self.transform:#如果没有transform的话就不变换
            img = self.transform(img) #trasform的参数是图象类型
        label = 0#测试集不需要label
        return img,label
test_data = TestDataset(test_list,transform=transform)
test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

# %%
# print(train_data.__len__())


class CNN(nn.Module):
    def __init__(self,num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.Dropout(0.5),
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )


        self.fc = nn.Sequential(
            nn.Linear(32 * 32 *32, out_features=128),
            nn.Linear(128, out_features=32),
            nn.Linear(in_features=32, out_features=num_classes)
        )
        
           # fully connected layer, output 5 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.fc(x)
        return output    # return x for visualization


# %%
import netron
x = torch.randn(BATCH_SIZE, 3, 128, 128)# 随机生成一个数据
my_Net = CNN()
model_path = './model.pth'
torch.onnx.export(my_Net, x, model_path)  # 将 pytorch 模型以 onnx 格式导出并保存
netron.start(model_path) # 输出网络结构

# %%
print(CNN())
# %%
def train():
    torch.cuda.empty_cache()
    torch.cuda.set_device(3)
    model = CNN().to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),lr=LR)#传入模型参数和学习率
    loss_list = []
    acc = []
    T1 = time.time()
    for i in range(EPOCH):
        for j,(batch_data,batch_label) in enumerate(train_loader):
            batch_data, batch_label = batch_data.cuda(),batch_label.cuda()#数据进行.cuda()处理。就可以将内存中的数据复制到GPU的显存中去
            #梯度清零
            optimizer.zero_grad()
            #前向传播
            output = model(batch_data)
            #计算损失
            loss = loss_fun(output,batch_label)
            loss_list.append(loss.item())
            #反向传播
            loss.backward()
            #更新权重
            optimizer.step()
        acc.append(test(model))
        
        print("Epoch[{}/{}],Loss:{:4f}".format(i+1,EPOCH,loss.item()))
        run_time = time.time() - T1
        hour = run_time//3600
        minute = (run_time-3600*hour)//60
        second = run_time-3600*hour-60*minute
        print (f'运行时间：{hour}小时{minute}分钟{second}秒')
    plt.plot(loss_list)
    plt.show()
    
def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():#反向传播时就不会自动求导了，因此大大节约了显存
        for j,(batch_data,batch_label) in enumerate(test_loader):
            print(len(batch_data))
            batch_data, batch_label = batch_data.to(device),batch_label.to(device)
            outputs = model(batch_data)
            labels = [train_data.class_to_idx[os.path.basename(f).rsplit('0', 1)[0]] for f in test_list]
            labels = torch.tensor(labels).to(device)
            length = len(labels)
            #labels = [train_data.class_to_idx[os.path.basename(f).rsplit('0', 1)[0]] for f in test_list]
            
            temp_labels = labels[j*BATCH_SIZE:min((j+1)*BATCH_SIZE,length)]
            temp_len=min((j+1)*BATCH_SIZE,length)-j*BATCH_SIZE
            total += temp_len
            print("outputs",outputs)
            predict = torch.Tensor.argmax(outputs.data,axis=1)
            #print(np.argmax(a, axis=0))  #竖着比较，返回行号
            #print(np.argmax(a, axis=1))  #横着比较，返回列号
            print("tmp_labels",temp_labels)
            correct += (predict==temp_labels).sum()

    print("Test Accuracy %.2f%%"% (100*correct/total))
    return correct/total

train()

# %%



