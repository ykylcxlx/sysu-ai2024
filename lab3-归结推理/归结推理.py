#!/usr/bin/env python
# coding: utf-8

# In[3]:


def convert_string_to_set(KB_string):
    KB_set = set()
    # 删除所有空格和大括号
    KB_string = KB_string.replace(" ", "").strip("{}")
    #print("KB_string = ",KB_string)
    # 按照逗号分割子句
    clauses = KB_string.split("),(")
    
    # 处理首个子句的左括号
    clauses[0] = clauses[0].lstrip("(")

    # 处理末尾子句的右括号
    clauses[-1] = clauses[-1].rstrip(")")
    if clauses[-1][-1]==',':
        clauses[-1]=clauses[-1][:-1]
    #去除了多个），重新加上一个
    else:
        clauses[-1]+=")"
    #print("clauses = ",clauses)
    # 遍历每个子句并将其转换为元组并添加到集合中
    for clause in clauses:
        clause = [x for x in clause.split(",") if x]#去除空字符
        #clause.split(',')
        #print("splitclause=",clause)
        new_clause = []
        i = 0
        #for i in range(len(clause)):
        while i < len(clause):
            if i<len(clause)-1 and clause[i+1][0].islower() and clause[i][-1].islower():
                new_clause.append(clause[i]+','+clause[i+1])
                i  += 2
            else:
                new_clause.append(clause[i])
                i += 1
        #print(new_clause)
        KB_set.add(tuple(new_clause))

    return KB_set

# # 示例输入
# KB_string = " {(A(tony),),(A(mike),),(A(john),),(L(tony,rain),),(L(tony,snow),),\
# (~A(x),S(x),C(x)),(~C(y),~L(y,rain)),(L(z,snow),~S(z)),\
# (~L(tony,u),~L(mike,u)),(L(tony,v),L(mike,v)),(~A(w),~C(w),S(w))}"
# # 转换为集合
# KB_set = convert_string_to_set(KB_string)
# # 打印结果
# print(KB_set)


# In[4]:


def indexfind(clause1,clause2):
    for i,literal1 in enumerate(clause1):
        if not literal1.startswith("~"):
            for j,literal2 in enumerate(clause2):
                if literal2.startswith("~"):
                    qianzhui1 = literal1.split("(")[0]
                    qianzhui2 = literal2.split("(")[0].lstrip('~')
                    if(qianzhui1 == qianzhui2):
                        return [i,j]
    for i,literal1 in enumerate(clause1):
        if literal1.startswith("~"):
            for j,literal2 in enumerate(clause2):
                if not literal2.startswith("~"):
                    qianzhui1 = literal1.split("(")[0].lstrip('~')
                    qianzhui2 = literal2.split("(")[0]
                    if(qianzhui1 == qianzhui2):
                        return [i,j]
    return [-1,-1]
clause1 = ('~Student(x)',)#注意只含一个元素的元组是（，）
clause2 =  ('Student(b,x)', 'HardWorker(x)')

indexfind(clause1,clause2)


# In[5]:


def tihuan(clause,substitution):
    var_list = clause.split("(")[1].split(")")[0].split(",")
    for var in var_list:
        if var in substitution:
            clause = clause.replace("("+var+")", "("+substitution[var]+")")
            clause = clause.replace("("+var+",", "("+substitution[var]+",")
            clause = clause.replace(","+var+")", ","+substitution[var]+")")
            clause = clause.replace(","+var+",", ","+substitution[var]+",")
    return clause
#print(tihuan("Student(x,y,z)",{"x":"a","z":"c","y":"b"}))

def is_variable(s):
    #return s.islower() and s.isalpha()
    return s in ['x','y','z','xx','yy','zz','u','v','w','uu','vv','ww']

def unify(var, x, substitution):
    if var in substitution:
        #return unify(substitution[var], x, substitution)
        if substitution[var] == x:
            return substitution
        else:
            return None
    elif x in substitution:
        #return unify(var, substitution[x], substitution)
        if substitution[x] == var:
            return substitution
        else:
            return None
    if is_variable(var) and is_variable(x) and var != x:
        substitution[var] = x
        return substitution
    if is_variable(var) and not is_variable(x) :
        substitution[var] = x
        return substitution
    elif is_variable(x) and not is_variable(var) :
        substitution[x] = var
        return substitution
    elif var == x :
        return substitution
    else:
        return None

# In[6]:

def resolve(clause1, clause2):
    resolved_clause = []
    substitution = {}
    flag = 0
    # 检查第一个子句中的正文字和第二个子句中的负文字
    i,j = indexfind(clause1,clause2)
    if i!= -1 and j!= -1:
        literal1 = clause1[i]
        literal2 = clause2[j]

        var_list = literal1.split("(")[1].split(")")[0].split(",")
        x_list = literal2.split("(")[1].split(")")[0].split(",")
        #var_list = var.split(",")
        for var,x in zip(var_list,x_list):
            if unify(var, x, substitution)==None:
                flag = 1
                return None,None
            if substitution!=None and unify(var, x, substitution)!=None:
                #if var not in substitution.values:
                substitution.update(unify(var, x, substitution))

        #if substitution is not None:
            
            # 创建新的子句，包含两个子句中除了正文字和负文字之外的所有文字
        for l in clause1:
            if l != literal1:
                resolved_clause.append(tihuan(l,substitution))
        for l in clause2:
            if l != literal2:
                resolved_clause.append(tihuan(l,substitution))
        resolved_clause = list(set(resolved_clause))
        if tuple(resolved_clause) != None:
            return tuple(resolved_clause),substitution
    #return tuple(resolved_clause),substitution
    
    return None,None

# 示例调用

clause1 = ('On(x,john)','str(x)')
clause2 = ('~On(tony,y)',)
resolved = resolve(clause1, clause2)
#print(resolved)


# In[7]:


class Forest():
    def __init__(self, clause=None, pre1=None, pre2=None, id1 = None ,id2 = None,substitution = None):
        self.clause = clause
        self.pre1 = pre1
        self.pre2 = pre2
        self.id1 = id1
        self.id2 = id2
        self.substitution = substitution


# In[41]:


def to_list(forest,clauses):
    relavent = [forest[-1]]
    
    for node in relavent:
        if node.pre1 and node.pre1 not in relavent and node.pre1.clause not in clauses:
            relavent.append(node.pre1)
        if node.pre2 and node.pre2 not in relavent and node.pre2.clause not in clauses:
            relavent.append(node.pre2) 
    relavent.reverse()
    # for i in relavent:
    #     print(i.clause)
    return relavent


# In[40]:
def ans_print(lst,clauses):
    print_lst = []
    cnt =len(clauses)+1
    ind1 = 0
    for i in lst:
        clauses.append(i.clause)
    j = 0
    for item in lst:
        print_str = f"{cnt} R[{clauses.index(item.pre1.clause)+1}"
        if item.id1!=None:
            print_str+=chr(item.id1+97)
        print_str += f",{clauses.index(item.pre2.clause)+1}"
        if item.id2!=None:
            print_str+=chr(item.id2+97)
        print_str += "]"
        if item.substitution != None and item.substitution != {}:
            print_str+='{'
            for k,v in item.substitution.items():
                if k == list(item.substitution.keys())[0]:
                    print_str += f"{k}={v}"
                else:
                    print_str += f",{k}={v}"
            print_str += '}'
        print_str += f" = {item.clause}"
        print(print_str)
        cnt +=1
        print_lst.append(print_str)
    return print_lst


# In[42]:


def resolve_clauses(clauses):
    ##clauses = list(clauses)
    key = ""
    value = ""

    # 存储归结步骤的列表
    resolution_steps = []
    count = len(clauses)+1
    # 初始化步骤编号
    step_number = 1
    # 复制子句以进行归结
    forest_list = []
    for i in clauses:
        forest_list.append(Forest(clause=i))
    resolved_clauses = clauses.copy()
    # 归结循环
    while True:
        flag = 0
        new_clause = None
        #print(resolved_clauses)
        # 遍历每对子句进行归结
        for i in range(len(resolved_clauses)):
            for j in range(i + 1, len(resolved_clauses)):
                #print("resolved_clauses[i], resolved_clauses[j]",resolved_clauses[i],resolved_clauses[j])
                a,b = indexfind(resolved_clauses[i], resolved_clauses[j])
                
                resolved = resolve(resolved_clauses[i], resolved_clauses[j])[0]
                substitution = resolve(resolved_clauses[i], resolved_clauses[j])[1]
                if substitution!={} and substitution!=None:

                    keys = list(substitution.keys())
                    values = list(substitution.values())

                # 如果归结结果为空子句，则完成归结
                if resolved==():
                    
                    #print(f"R[{i+1},{j+1}] = []")
                    forest_list.append(Forest(resolved,forest_list[i],forest_list[j],
                                              id1 = a if len(resolved_clauses[i])>1 else None,
                                              id2 = b if len(resolved_clauses[j])>1 else None,
                                              substitution = substitution))
                    to_list(forest_list,clauses)
                    ans_print(to_list(forest_list,clauses),clauses)
                    resolution_steps.append(f"R[{i+1},{j+1}] = []")
                    return resolution_steps

                # 如果归结结果不为空且未在之前的步骤中出现，则进行记录
                if resolved and resolved not in resolved_clauses and 1 :#resolved != new_clause
                    new_clause = resolved
                    forest_list.append(Forest(resolved,forest_list[i],forest_list[j],
                                              id1 = a if len(resolved_clauses[i])>1 else None,
                                              id2 = b if len(resolved_clauses[j])>1 else None,
                                              substitution =substitution))
                    resolved_clauses.append(new_clause)
                    #print(resolved_clauses)
                    # print_str = f"R[{i+1}"
                    # if len(resolved_clauses[i])>1 :
                    #     print_str += chr(a+97)
                    # print_str += f",{j+1}"
                    # if len(resolved_clauses[j])>1 :
                    #     print_str += chr(b+97)
                    # print_str += "]"
                    # if substitution != None:
                    #     for k,v in substitution.items():
                    #         print_str += f"{{{k}={v}}}"
                    #     print_str += f" = {resolved}"
                    #     #resolution_steps.append(f"{cnt} R[{i+1},{j+1}]{{{key}={value}}} = {resolved}")
                    
                    # #print(count,print_str)
                    
                    # resolution_steps.append(print_str)
                    # count += 1
                if i == len(resolved_clauses) - 2 and j == len(resolved_clauses) - 1 and not resolved:
                    print("不能归结出空子句")
                    return None

                    #else:
                        #resolution_steps.append(f"{cnt} R[{i+1},{j+1}] = {resolved}")
                        #cnt +=1

        # 如果没有新的归结子句产生，则无法进一步归结，退出循环
        # if not new_clause:
        #     break

        # 添加新归结子句到已归结子句列表中
        #resolved_clauses.append(new_clause)

        # 添加归结步骤到列表中
        #resolution_steps.append(f"{step_number} {new_clause}")

        # 更新步骤编号
        #step_number += 1


# 示例输入


# In[26]:
def solve(KBstring):

    KB = convert_string_to_set(KBstring)
    clauses = list(KB)
    cnt = 1
    for item in clauses:
        print(cnt,item)
        cnt += 1
    # 执行归结推理并获取归结步骤列表
    resolution_steps = resolve_clauses(clauses)
    #print(resolution_steps)
if __name__ == "__main__":
    KB_str0 = "{(GradStudent(sue),),(~GradStudent(x),Student(x)),(~Student(x),HardWorker(x)),(~HardWorker(sue),)}"
    KB_str1 = "{(A(tony),),(A(mike),),(A(john),),(L(tony,rain),),(L(tony,snow),),(~A(x),S(x),C(x)),(~C(y),~L(y,rain)),(L(z,snow),~S(z)),(~L(tony,u),~L(mike,u)),(L(tony,v),L(mike,v)),(~A(w),~C(w),S(w))}"
    KB_str2 = "{(On(tony,mike),),(On(mike,john),),(Green(tony),),(~Green(john),),(~On(xx,yy),~Green(xx),Green(yy))}"
    
    my_str0 = "{(A(fuck),),(B(x),C(x),~A(x)),(~B(fuck),),(~C(fuck),)}"
    print("\n==============TEST0================\n")
    result = solve(KB_str0)
    # result = ResolutionFOL(KB_str0, full_print=True) # 打印完整推理过程
    #print("\n".join(result))
    
    print("\n==============TEST1================\n")
    result = solve(KB_str1)
    #print("\n".join(result))
    
    print("\n==============TEST2================\n")
    result = solve(KB_str2)
    #print("\n".join(result))

# In[36]:

