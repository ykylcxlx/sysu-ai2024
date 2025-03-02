import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import heapq
import time
class GeneticAlgTSP:
    def __init__(self, filename,population_size=60,mutate_rate=0.05):
        # 读取数据
        self.filename = filename[:-4]
        #citis： np array (num_of_cities, 2) 
        self.cities = self.read_tsp_data(filename)
        self.num_cities = len(self.cities)
        # 邻接矩阵
        self.distance_map = np.array([[0 for _ in range(self.num_cities)] for _ in range(self.num_cities)])
        # 初始化种群
        self.population = self.initialize_population(pop_size=100)
        self.population_size = population_size
        self.mutation_rate = mutate_rate#这里后续没有使用到
        self.solution = [i for i in range(len(self.cities))]
        self.dis_of_iter = []
    def read_tsp_data(self, filename):
        # 读取TSP数据文件的实现
        cities = []
        with open(filename, 'r') as f:
            for line in f:
                if "NODE_COORD_SECTION" in line:
                    break
            for line in f:
                if "EOF" in line:
                    break
                parts = line.strip().split()
                city_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                cities.append([x, y])
            #print(cities)
        return np.array(cities)
    def get_distance_map(self):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.distance_map[i][j] = self.distance(self.cities[i],self.cities[j])
        return self.distance_map
    def distance(self,v1:np.array,v2:np.array):
        return np.linalg.norm(v1-v2)
    def initialize_population(self, pop_size):
        # 初始化种群的实现
        population = []
        for _ in range(2*pop_size):
            chromosome = list(range(self.num_cities))
            random.shuffle(chromosome)
            population.append(chromosome)
            #print(population)
        #self.population = self.rank(population)
        return population[0:pop_size//2]
    
    def fitness(self, chromosome):
        # 计算适应度的实现
        total_distance = 0
        for i in range(self.num_cities):
            city_a = chromosome[i]
            city_b = chromosome[(i + 1) % self.num_cities]#注意最后要形成回路
            total_distance += self.distance_map[city_a][city_b]
            if(total_distance==0):#防止除0错误
                return 10
        return 1 / total_distance
        
    def best(self,lst):
        temp_best = lst[0]
        for it in lst:
            if self.fitness(it)>self.fitness(temp_best):
                temp_best = it
        return temp_best
    import heapq

    def get_smallest_elements(self,nums, lst):
        # 使用堆来获取前K小的元素
        smallest_elements = heapq.nlargest(nums, lst, key=self.fitness)
        #smallest_elements.sort()  # 可选步骤，按升序排序
        return smallest_elements
    def select(self,father_and_son):
        # 锦标赛
        top_n = self.population_size//10
        #self.population = self.get_smallest_elements(top_n, self.population)
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = self.num_cities // group_num  # 每小组获胜人数
        winners = self.get_smallest_elements(top_n, father_and_son)  # 先选取前10%的个体
        # for i in range(group_num):
        #     group = []
        #     for j in range(group_size):
        #         # 随机组成小组
        #         player = random.choice(self.population)
        #         group.append(player)
        #     group = self.rank(group)
        #     # 取出获胜者
        #     winners += group[:group_winner]
        for i in range(self.population_size-top_n):
            group = []
            for j in range(group_size):
                group.append(random.choice(father_and_son[top_n:]))
            winners.append(self.best(group))

        self.population = winners
    #@staticmethod
    def rank(self,group):
        # 冒泡排序 （没被使用到）
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if self.fitness(group[j+1]) > self.fitness(group[j]):
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group
    
    def select_parents(self):
        # 选择父母的实现
        fitness_scores = [self.fitness(chromosome) for chromosome in self.population]#轮盘赌算法
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]
        #print("probabilities = ",probabilities)
        parents = []
        for _ in range(self.population_size):
            parent = random.choices(self.population, probabilities)[0]
            #print(random.choices(self.population, probabilities))
            parents.append(parent)
        return parents
    def crossover(self, parent1, parent2):
        # 交叉操作的实现
        start_index = random.randint(0, self.num_cities - 2)# 随机返回[a,b]区间内的一个整数
        end_index = random.randint(start_index, self.num_cities - 1)
        child1 = [None] * self.num_cities
        child1[start_index:end_index+1] = parent1[start_index:end_index+1]
        #子代start[start_index:end_index+1]与parent1相同
        for gene in parent2:
            if gene not in child1:
                for i, c in enumerate(child1):
                    if c is None:
                        child1[i] = gene
                        break
        child2 = [None] * self.num_cities
        child2[start_index:end_index+1] = parent2[start_index:end_index+1]
        for gene in parent2:
            if gene not in child2:
                for i, c in enumerate(child2):
                    if c is None:
                        child2[i] = gene
                        break
        return [child1,child2]
    
    # def mutate(self, chromosome, mutation_rate):
    #     # 变异操作的实现
    #     for i in range(self.num_cities):
    #         if random.random() < mutation_rate:
    #             j = random.randint(0, self.num_cities - 1)

    #             chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    #     return chromosome
        #倒置变异
    def mutate(self,way): #变异函数
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
    
    def iterate(self, num_iterations):
        # 迭代过程的实现
        plt.ion()
        
        T1 = time.time()
        for i in range(num_iterations):
            new_population = []
            parents = self.select_parents()
            for _ in range(0, self.population_size, 2):
                parent1, parent2 = random.sample(parents, 2)
                childs = self.crossover(parent1, parent2)
                for child in childs:
                    child = self.mutate(child,)
                    new_population.append(child)
            father_and_son = self.population+new_population
            self.select(father_and_son)
            #self.population = new_population
            self.solution = self.best(new_population)
            self.dis_of_iter.append(1/self.fitness(self.solution))

            self.dynamic_draw()
            print("第{}次迭代的distance{}".format(i, 1/self.fitness(self.solution)))
            #print("pop1",1/self.fitness(self.population[0]),"pop2",1/self.fitness(self.population[1]))
            # if 1/self.fitness(self.population[0])< 1/6* sum(self.distance_map[0]):
            #     break

      
        # 找到最佳解
        best_fitness = 0
        best_solution = None
        for chromosome in self.population:
            fitness_score = self.fitness(chromosome)
            if fitness_score > best_fitness:
                best_fitness = fitness_score
                best_solution = chromosome
        self.solution = best_solution
        T1 = time.time()
        run_time = time.time() - T1
        hour = run_time//3600
        minute = (run_time-3600*hour)//60
        second = run_time-3600*hour-60*minute
        print (f'该程序运行时间：{hour}小时{minute}分钟{second}秒')
        self.draw_dis()


        return best_solution
    def dynamic_draw(self):
        plt.clf()
        cities = copy.deepcopy(self.cities)
        x = []
        y = []
        for it in self.solution:
            x.append(cities[it][0])
            y.append(cities[it][1])
        x.append(x[0])
        y.append(y[0])
        plt.plot(x,y,'g--',markersize=3,linewidth=1)#,marker='o'
        plt.scatter(x,y,c='r',marker='*',s=20,linewidth=1)
        plt.show()
        plt.pause(0.05)

    def draw(self):
        plt.clf()
        cities = copy.deepcopy(self.cities)
        #print(np.array([cities[0]]))
        #cities.insert(cities[0],axis = 0)
        #cities = np.append(cities,np.array([cities[0]]),axis=0)
        x = []
        y = []
        for it in self.solution:
            x.append(cities[it][0])
            y.append(cities[it][1])
        x.append(x[0])
        y.append(y[0])
        plt.plot(x,y,'g--',markersize=3,linewidth=1)#,marker='o'
        plt.scatter(x,y,c='r',marker='*',s=20,linewidth=1)
        figname = 'figs/' + self.filename+ '_path'
        plt.savefig(figname)
    def draw_dis(self):
        plt.clf()
        plt.plot([i for i in range(len(self.dis_of_iter))],self.dis_of_iter)
        figname = 'figs/' + self.filename +'_fitness'
        plt.savefig(figname)

# # 使用示例

tsp_solver = GeneticAlgTSP("dj38.tsp",200)
tsp_solver.get_distance_map()
#print("distance_map",tsp_solver.distance_map)#为什么用法get返回的是一个《》
print("origin distance:",1/tsp_solver.fitness(tsp_solver.population[0]))
tsp_solver.draw()
best_solution = tsp_solver.iterate(300)
print("solution",best_solution)
print("diatance",1/tsp_solver.fitness(best_solution))
tsp_solver.draw()
plt.ioff()
plt.show()
# for i,it in enumerate(best_solution):
#     if i < len(best_solution)-1:
#         plt.plot([tsp_solver.cities[best_solution[i]][0],tsp_solver.cities[best_solution[i+1]][0]],[tsp_solver.cities[best_solution[i]][1],tsp_solver.cities[best_solution[i+1]][1]],'g--',marker='o',markersize=3,linewidth=1)
        #plt.scatter([tsp_solver.cities[i][0],tsp_solver.cities[i+1][0]],[tsp_solver.cities[i][1],tsp_solver.cities[i+1][1]],'g--',marker='o',markersize=3,linewidth=1)
# plt.plot(x1[1] y1,'g--',marker='o',markersize=3,linewidth=1) 
