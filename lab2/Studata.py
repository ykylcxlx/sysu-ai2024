
import time

class StuData():
    def __init__(self,path):
        with open(path,'r',encoding='gbk') as f:
            data = f.read().splitlines()
            #data = f.readlines()
        data = [i.strip().split() for i in data]#strip 用于移除字符串中的空格、换行(\n)、制表符(\t)，split 用于拆分字符串
        for i in range(len(data)):
            data[i][3] = int(data[i][3])
        f.close()
        self.data = data
    def AddData(self,name,stu_num,gender,age):
        return  self.data.append([name,stu_num,gender,age])#在列表末尾添加新的列表
    def SortData(self,s):
        self.data = sorted(self.data, key=lambda x:x[dct[s]])#int
        return self.data
    def ExportFile(self,path):
        with open(path,'w',encoding='gbk') as f:
           for i in self.data:
                f.write(i[0]+' '+i[1]+' '+i[2]+' '+str(i[3])+' '+'\n')
           f.close()
                #x=str(i)
                #f.writelines(x)
if __name__ == "__main__":
    dct = {"name":0,"stu_num":1,"gender":2,"age":3}
    path = r"student_data.txt"#注意相对路径的写法
    time_start = time.time()  # 记录开始时间
    s = StuData(path)
    s.AddData("Tom","246","M",18)
    s.SortData("stu_num")
    s.ExportFile('new_stu_data.txt')
    print(s.data)
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)