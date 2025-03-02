#遗传算法
import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data=[]
x=[]
y=[]

def read(file):
    ans=[]
    with open(file,'r') as file_object:
        for line in file_object.readlines():
            temp=line.strip()
            temp=temp.split()
            t=[int(temp[0]),float(temp[1]),float(temp[2])]
            x.append(float(temp[1]))
            y.append(float(temp[2]))
            ans.append(t)
    return ans

def get_dist(way): #计算距离总和 算评估函数
    ans=0
    for i in range(1,len(way)+1):
        if i==len(way):
            way1=way[0]-1
            way2=way[len(way)-1]-1
        else:
            way1=way[i-1]-1
            way2=way[i]-1
        x1=data[way1][1]
        y1=data[way1][2]
        x2=data[way2][1]
        y2=data[way2][2]
        ans+=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return ans

def n1(way):
    p=random.random()
    a,b,c=random.sample(range(1, len(way)-1), 3)
    point=sorted([a,b,c]) #升序
    a,b,c=point[0],point[1],point[2]
    if p<0.5: #交换中间两段
        ans=way[0:a]+way[b:c]+way[a:b]+way[c:len(way)]
    else: #交换首尾两段
        ans=way[c:len(way)]+way[a:b]+way[b:c]+way[0:a]
    return ans

def n2(way):
    i=random.randint(0,len(way)-2)
    j=random.randint(1,len(way)-1)
    while True:
        if i!=j: #如果不相同 则交换两个城市
            way[i],way[j]=way[j],way[i]
            ans=way[:]
            way[i],way[j]=way[j],way[i]
            break
        else:
            i=random.randint(0,len(way)-2)
            j=random.randint(1,len(way)-1)
    return ans

def n3(way):
    a,b=random.sample(range(1, len(way)-1), 2)
    if a>b: a,b=b,a 
    ans=way[0:a]+way[a:b][::-1]+way[b:len(way)] #中间反转
    return ans

def get_neighbur(way): #变异函数
    p=random.random()
    if p<0.5:#分成四段
        a,b,c=random.sample(range(1, len(way)-1), 3)
        point=sorted([a,b,c]) #升序
        a,b,c=point[0],point[1],point[2]
        if p<0.25: #交换中间两段
            ans=way[0:a]+way[b:c]+way[a:b]+way[c:len(way)]
        else: #交换首尾两段
            ans=way[c:len(way)]+way[a:b]+way[b:c]+way[0:a]

    elif p<0.75: #选择其中两个点
        i=random.randint(0,len(way)-2)
        j=random.randint(1,len(way)-1)
        while True:
            if i!=j: #如果不相同 则交换两个城市
                way[i],way[j]=way[j],way[i]
                ans=way[:]
                way[i],way[j]=way[j],way[i]
                break
            else:
                i=random.randint(0,len(way)-2)
                j=random.randint(1,len(way)-1)

    else:#分成三段
        a,b=random.sample(range(1, len(way)-1), 2)
        if a>b: a,b=b,a 
        ans=way[0:a]+way[a:b][::-1]+way[b:len(way)] #中间反转
    return ans

def first_group(way): #生成若干个初始样例 选择其中最好的一部分并作为初始种群返回
    population=[]
    for i in range(0,50):
        temp_way=way[:]
        random.shuffle(temp_way)
        if temp_way not in population:
            population.append(temp_way)
    ans=sorted(population,key=lambda x:get_dist(x)) 
    return ans[:4] #返回前4个距离最小的

def get_all_p(population,length):
    dist=[]
    sum=0
    for i in range(length):
        a=get_dist(population[i])
        dist.append(a)
        sum+=a
    ans=[]
    for i in range(length):
        a=1-(dist[i]/sum)
        ans.append(a)
    return dist,ans

def get_kid(dad,mom):
    ans=[0 for i in range(len(dad))]
    a,b=random.sample(range(1, len(dad)-1), 2)
    if a>b: a,b=b,a 
    for i in range(a,b): ans[i]=dad[i]
    i=j=0
    for i in range(0,len(mom)):
        if mom[i] not in dad[a:b]:
            ans[j]=mom[i]
            j+=1
        if j==a: j=b
    return ans

if __name__ == '__main__':
    file="temp1.txt"
    data=read(file)#每一行数据写入到list中

    way=[i for i in range(1,len(data)+1)] #旅行商走过的路径
    distance=get_dist(way)
    population=first_group(way)   #生成初始种群
    p=0.333 #变异概率

    record_weight=[] #保存每次最优的距离 呈现变换
    fig1=plt.figure(1)
    gif_pic=[] #保存动态图
    
    best_way=population[0]
    best_dist=get_dist(best_way)
    sum=0 #迭代次数
    while sum<250000:
        sum+=1
        new_population=[]
        length=len(population)//2
        for i in range(length):
            #选择父母
            all_weight,all_p=get_all_p(population,len(population)) #群体中每个个体被选择的概率
            dad=random.choices(population,all_p,k=1) #轮盘赌
            mom=random.choices(population,all_p,k=1)
            #生成子代
            kid1=get_kid(dad[0],mom[0])
            kid2=get_kid(dad[0],mom[0])
            #是否变异 且必须不相同
            while True:
                if random.random()<p or kid1==kid2:
                    # kid1=n1(kid1)
                    # kid2=n1(kid2)
                    # kid1=n2(kid1)
                    # kid2=n2(kid2)
                    # kid1=n3(kid1)
                    # kid2=n3(kid2)
                    kid1=get_neighbur(kid1)
                    kid2=get_neighbur(kid2)
                if kid1!=kid2:
                    break
            new_population.append(kid1)
            new_population.append(kid2)
        i=j=0
        best_now=9999
        for i in range(len(population)):
            if all_weight[i]<best_now:
                way=population[i]
                best_now=all_weight[i] #每次选出最好的一个个体 放入下一个种群中
        record_weight.append(best_now)
        if best_now<best_dist:
            best_dist=best_now
            best_way=way
        population=new_population[:-1]
        population.append(way)

        if sum%100==0:
            x=[]
            y=[]
            for i in way:
                x.append(data[i-1][1])
                y.append(data[i-1][2])
            x.append(data[0][1])
            y.append(data[0][2]) 
            plt.title('process')
            pic=plt.plot(x, y, marker = '.', color = 'blue',linewidth=1) 
            gif_pic.append(pic)

    print(best_way)
    print(best_dist)

    # fig2 = plt.figure(2) #最优解
    # x=[]
    # y=[]
    # for var in way:
    #     x.append(data[var-1][1])
    #     y.append(data[var-1][2])
    # x.append(data[0][1])
    # y.append(data[0][2])
    # plt.title('Final solution of TSP')
    # plt.plot(x,y,marker='.',color ='red',linewidth=1) 
    # gif_pic.append(pic)

    #GIF保存
    ani=animation.ArtistAnimation(fig1,gif_pic,interval=200, repeat_delay=1000)
    ani.save("GA_TSP.gif",writer='pillow')

    fig2=plt.figure(2) #展现最优距离的变化
    plt.title('cost change')
    x_=[i for i in range(len(record_weight))] #横轴
    plt.plot(x_,record_weight) #横轴纵轴
    plt.show()