
def MatrixAdd(A, B):
   
    C = [ [0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] + B[i][j]
    return C


def MatrixMul(A, B):
    if(len(A[0])!=len(B)):
        print("不能相乘")
        return None
    C = [ [0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range (len(B)):
                C[i][j] += A[i][k] * B[k][j]#k
    return C


if __name__ == "__main__":
    r = c = 3
    A = [[1,2,3],[4,5,6],[7,8,9]]
    B = [[1,2,3],[4,5,6],[7,8,9]]
    #l1=[[9,9,9],[9,9,9]]
    #l2=[[1,1],[1,1],[1,1]]
    print(MatrixAdd(A, B))
    print(MatrixMul(A, B))
    #print(MatrixMul(l1, l2))