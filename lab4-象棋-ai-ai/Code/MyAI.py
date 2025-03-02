import copy
from ChessBoard import *
count = 1
from Game import *
class Evaluate(object):
    # 棋子棋力得分
    single_chess_point = {
        'c': 989,   # 车
        'm': 439,   # 马
        'p': 542,   # 炮
        's': 226,   # 士
        'x': 210,   # 象
        'z': 55,    # 卒
        'j': 65536  # 将
    }
    # 红兵（卒）位置得分
    red_bin_pos_point = [
        [1, 3, 9, 10, 12, 10, 9, 3, 1],
        [18, 36, 56, 95, 118, 95, 56, 36, 18],
        [15, 28, 42, 73, 80, 73, 42, 28, 15],
        [13, 22, 30, 42, 52, 42, 30, 22, 13],
        [8, 17, 18, 21, 26, 21, 18, 17, 8],
        [3, 0, 7, 0, 8, 0, 7, 0, 3],
        [-1, 0, -3, 0, 3, 0, -3, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # 红车位置得分
    red_che_pos_point = [
        [185, 195, 190, 210, 220, 210, 190, 195, 185],
        [185, 203, 198, 230, 245, 230, 198, 203, 185],
        [180, 198, 190, 215, 225, 215, 190, 198, 180],
        [180, 200, 195, 220, 230, 220, 195, 200, 180],
        [180, 190, 180, 205, 225, 205, 180, 190, 180],
        [155, 185, 172, 215, 215, 215, 172, 185, 155],
        [110, 148, 135, 185, 190, 185, 135, 148, 110],
        [100, 115, 105, 140, 135, 140, 105, 115, 110],
        [115, 95, 100, 155, 115, 155, 100, 95, 115],
        [20, 120, 105, 140, 115, 150, 105, 120, 20]
    ]
    # 红马位置得分
    red_ma_pos_point = [
        [80, 105, 135, 120, 80, 120, 135, 105, 80],
        [80, 115, 200, 135, 105, 135, 200, 115, 80],
        [120, 125, 135, 150, 145, 150, 135, 125, 120],
        [105, 175, 145, 175, 150, 175, 145, 175, 105],
        [90, 135, 125, 145, 135, 145, 125, 135, 90],
        [80, 120, 135, 125, 120, 125, 135, 120, 80],
        [45, 90, 105, 190, 110, 90, 105, 90, 45],
        [80, 45, 105, 105, 80, 105, 105, 45, 80],
        [20, 45, 80, 80, -10, 80, 80, 45, 20],
        [20, -20, 20, 20, 20, 20, 20, -20, 20]
    ]
    # 红炮位置得分
    red_pao_pos_point = [
        [190, 180, 190, 70, 10, 70, 190, 180, 190],
        [70, 120, 100, 90, 150, 90, 100, 120, 70],
        [70, 90, 80, 90, 200, 90, 80, 90, 70],
        [60, 80, 60, 50, 210, 50, 60, 80, 60],
        [90, 50, 90, 70, 220, 70, 90, 50, 90],
        [120, 70, 100, 60, 230, 60, 100, 70, 120],
        [10, 30, 10, 30, 120, 30, 10, 30, 10],
        [30, -20, 30, 20, 200, 20, 30, -20, 30],
        [30, 10, 30, 30, -10, 30, 30, 10, 30],
        [20, 20, 20, 20, -10, 20, 20, 20, 20]
    ]
    # 红将位置得分
    red_jiang_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 9750, 9800, 9750, 0, 0, 0],
        [0, 0, 0, 9900, 9900, 9900, 0, 0, 0],
        [0, 0, 0, 10000, 10000, 10000, 0, 0, 0],
    ]
    # 红相或士位置得分
    red_xiang_shi_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 60, 0, 0, 0, 60, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [80, 0, 0, 80, 90, 80, 0, 0, 80],
        [0, 0, 0, 0, 0, 120, 0, 0, 0],
        [0, 0, 70, 100, 0, 100, 70, 0, 0],
    ]

    red_pos_point = {
        'z': red_bin_pos_point,
        'm': red_ma_pos_point,
        'c': red_che_pos_point,
        'j': red_jiang_pos_point,
        'p': red_pao_pos_point,
        'x': red_xiang_shi_pos_point,
        's': red_xiang_shi_pos_point
    }

    def __init__(self, team):
        self.team = team

    def get_single_chess_point(self, chess: Chess):
        if chess.team == self.team:
            return self.single_chess_point[chess.name]
        else:
            return -1 * self.single_chess_point[chess.name]

    def get_chess_pos_point(self, chess: Chess):
        red_pos_point_table = self.red_pos_point[chess.name]
        if chess.team == 'r':
            pos_point = red_pos_point_table[chess.row][chess.col]
        else:
            pos_point = red_pos_point_table[9 - chess.row][chess.col]
        if chess.team != self.team:
            pos_point *= -1
        return pos_point

    def evaluate(self, chessboard: ChessBoard):
        point = 0
        for chess in chessboard.get_chess():
            point += self.get_single_chess_point(chess)
            point += self.get_chess_pos_point(chess)
        return point


class ChessMap(object):
    def __init__(self, chessboard: ChessBoard):
        self.chess_map = copy.deepcopy(chessboard.chessboard_map)


class MyAI(object):
    def __init__(self, user_team):
        self.team = user_team
        self.max_depth = 4
        self.old_pos = [0, 0]
        self.new_pos = [0, 0]
        self.evaluate_class = Evaluate(self.team)
        self.step = []
        self.repeat = 0

    def get_next_step(self, chessboard: ChessBoard):
        cur_step = self.old_pos + self.new_pos
  
            # 使用alpha-beta剪枝的minimax递归
        #score = self.minimax(chessboard, self.max_depth, -float('inf'), float('inf'))
        

        score = self.minimax(chessboard, 1, -float('inf'), float('inf'))

        if cur_step == self.old_pos + self.new_pos:
            # self.max_depth = 1
            # self.minimax(chessboard, 1, -float('inf'), float('inf'))
            for chess_line in chessboard.chessboard_map:
                for chess in chess_line:
                    if chess and chess.team == self.team:
                        move_position_list = chessboard.get_put_down_position(chess)
                        if move_position_list:  # 只要找到一个可以移动的位置，就表示没有失败，还是有机会的
                            return [chess.row,chess.col,move_position_list[0][0],move_position_list[0][1]]
            # print("lose")
        self.step.append(self.old_pos + self.new_pos)
        return self.old_pos + self.new_pos
    def minimax(self, chessboard:ChessBoard, depth, alpha, beta):
        if depth > self.max_depth or self.max_depth== 0 or chessboard.judge_win(self.team):
            #print("end")
            return self.evaluate_class.evaluate(chessboard)
        #获取当前所有棋子
        chesses = chessboard.get_chess()
        #print("chesses=",chesses)
        for chess in chesses:
            #max层
            if depth % 2 == 1 and chess.team == self.team:
                #max_score = -float('inf')
                for new_x,new_y in chessboard.get_put_down_position(chess):
                    #原来下一步上的棋：
                    ori_cs = chessboard.chessboard_map[new_x][new_y]
                    old_x,old_y = chess.row,chess.col
                    chessboard.my_move(chess, new_x, new_y)
                    # print ("my ai chesses=")
                    #print("assume")
                    # for cs in chessboard.get_chess():
                    #     print(cs.team, cs.name,cs.row, cs.col)
                    #chess.update_position(new_x, new_y)
                    score = self.minimax(chessboard, depth + 1, alpha, beta)#跟新depth,走到下一层
                    #print("score= ",score)
                    chessboard.my_move(chess,old_x,old_y)
                    #chess.update_position(old_x, old_y)
                    chessboard.chessboard_map[new_x][new_y]=ori_cs
                    #print("back")
                    # # 吃子
                    # if chessboard.chessboard_map[new_x][new_y]:
                    #     score += 1000
                    # # 将军
                    if chessboard.judge_attack_general(self.team):
                        score += 5000  # 将军
                    if len(self.step)>1:
                        if depth == 1 and [old_x,old_y,new_x, new_y] == self.step[-2]:
                            #print("有重复")
                            score -= 200
                        if depth == 1 and len(self.step)>3 and [old_x,old_y,new_x, new_y] == self.step[-2] and [old_x,old_y,new_x, new_y] == self.step[-4]:
                            print("两次重复")
                            score -= 80000
                        if len(self.step)> 3  and depth == 1 and [old_x,old_y,new_x, new_y] == self.step[-4]:
                            print("四步一个循环的重复")
                            score -= 1000
                        if len(self.step)> 7  and depth == 1 and [old_x,old_y,new_x, new_y] == self.step[-4] and [old_x,old_y,new_x, new_y] == self.step[-8]:
                            score -= 80000
                    

                            

                
                    if(score> alpha ) and depth==1:
                            #print("alpha,score",alpha,score)
                            self.old_pos=[chess.row,chess.col]
                            self.new_pos=[new_x,new_y]
                            #print("old_pos,new_pos",self.old_pos,self.new_pos)

                    alpha = max(alpha, score)
                    
                    if beta <= alpha:
                        return alpha
                #return score
            else:
                 if depth % 2 == 0 and chess.team != self.team:
                    #min_score = float('inf')
                    for new_x,new_y in chessboard.get_put_down_position(chess):
                        #原来下一步上的棋：
                        ori_cs = chessboard.chessboard_map[new_x][new_y]
                        old_x,old_y = chess.row,chess.col
                        chessboard.my_move(chess, new_x, new_y)
                        chess.update_position(new_x, new_y)
                        score = self.minimax(chessboard, depth + 1, alpha, beta)#跟新depth,走到下一层
                        #print("score= ",score)
                        #复原
                        chessboard.my_move(chess,old_x,old_y)
                        chessboard.chessboard_map[new_x][new_y]=ori_cs
                        #print("back")
                        #min_score = min(min_score, score)
                        beta = min(beta, score)
                        if beta <= alpha:
                            return beta
                    #return score
        if depth%2==1:
            return alpha
        else:
            return beta
   