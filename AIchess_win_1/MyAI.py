import copy
from ChessBoard import *
import random
import time

class TranspositionNode(object):
    def __init__(self, num_chesses, depth, score, move):
        self.num_chesses: int = num_chesses
        self.depth: int = depth
        self.score: int = score
        self.move: tuple[int] = move



class Evaluate(object):
    # 棋子棋力得分
    single_chess_point = {
        'c': 989,   # 车
        'm': 439,   # 马
        'p': 542,   # 炮
        's': 226,   # 士
        'x': 210,   # 象
        'z': 55,    # 卒
        'j': 1000000  # 将
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
        self.opponent = 'b' if team == 'r' else 'r'

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
        # 不要和棋/重复局面
        if chessboard.winner:
            if chessboard.winner == self.team:
                return 10000000
            else:
                return -10000000
        point = 5 # 先手优势
        for chess in chessboard.get_chess():
            point += self.get_single_chess_point(chess)
            point += self.get_chess_pos_point(chess)
        return point


def get_best_depth(num_legal_moves: int) -> int:
    """
    根据合法走法的数量，获取最佳的搜索深度。
    """
    return 8

class MyAI(object):
    def __init__(self, player):
        self.team = player
        self.opponent = 'b' if player == 'r' else 'r'
        self.evaluate_class = Evaluate(self.team)
        self.transposition_table_min: dict[int, TranspositionNode] = {}
        self.transposition_table_max: dict[int, TranspositionNode] = {}
        # 历史表, 4维数组
        self.history_table: dict[str, list[list[int]]] = {"r_c": None, "r_m": None, "r_p": None, "r_s": None, "r_x": None, "r_z": None, "r_j": None, 
                                                          "b_c": None, "b_m": None, "b_p": None, "b_s": None, "b_x": None, "b_z": None, "b_j": None}
        for key in self.history_table.keys():
            tuple(tuple(tuple(tuple(0 for _ in range(9)) for _ in range(10)) for _ in range(9)) for _ in range(10))
                
        
    def get_transposition_node(self, maximizingPlayer: bool, board_hash: int):
        if maximizingPlayer:
            return self.transposition_table_max.get(board_hash, None)
        else:
            return self.transposition_table_min.get(board_hash, None)
    
    def set_transposition_node(self, maximizingPlayer: bool, transposition_node: TranspositionNode, board_hash: int):
        if maximizingPlayer:
            self.transposition_table_max[board_hash] = transposition_node 
        else:
            self.transposition_table_min[board_hash] = transposition_node
    
    def delete_old_transposition_nodes(self, num_chesses: int):
        new_transposition_table_max = copy.deepcopy(self.transposition_table_max)
        new_transposition_table_min = copy.deepcopy(self.transposition_table_min)
        for key, item in self.transposition_table_max.items():
            if item.num_chesses > num_chesses:
                del new_transposition_table_max[key]
        for key, item in self.transposition_table_min.items():
            if item.num_chesses > num_chesses:
                del new_transposition_table_min[key]
        self.transposition_table_max = new_transposition_table_max
        self.transposition_table_min = new_transposition_table_min
        
    
    def evaluate_board(self, board: ChessBoard) -> int:
        """
        评估棋盘得分。
        """
        eval_obj = Evaluate(self.team)
        return eval_obj.evaluate(board)
    
    def get_legal_moves(self, board: ChessBoard, team: str) -> list:
        """
        获取合法的走法。
        """
        legal_moves = []
        for chess in board.get_chess():
            if chess.team == team:
                put_down_positions = board.get_put_down_position(chess, delete_general=True, none_when_end=True)
                for put_down_position in put_down_positions:
                    legal_moves.append((chess.row, chess.col, put_down_position[0], put_down_position[1]))
        return legal_moves
    
    
    def get_sorted_legal_moves(self, board: ChessBoard, team: str) -> list:
        """
        获取带有分数的合法走法。
        依次排序：置换表 > 将军 > 吃子 > 静态评估
        """
        legal_moves = []
        maxingPlayer = True if team == self.team else False
        for chess in board.get_chess():
            if chess.team == team:
                put_down_positions = board.get_put_down_position(chess, delete_general=True, none_when_end=True)
                for put_down_position in put_down_positions:
                    new_board = board.copy_smallest()
                    new_board.move_chess2(chess.row, chess.col, put_down_position[0], put_down_position[1])
                    score = 0
                    
                    # 置换表
                    transposition_node = self.get_transposition_node(maxingPlayer, new_board.map_hash())
                    if transposition_node:
                        score += transposition_node.score * 100 * transposition_node.depth * transposition_node.depth
                    # 吃子
                    if board.chessboard_map[put_down_position[0]][put_down_position[1]]:
                        score += 1000 * (1 if maxingPlayer else -1)
                    # 将军
                    if new_board.judge_attack_general(self.team):
                        score += 5000 * (1 if maxingPlayer else -1) # 将军
                    legal_moves.append((chess.row, chess.col, put_down_position[0], put_down_position[1], score))
                    # 静态评估
                    score += self.evaluate_board(new_board)
        random.shuffle(legal_moves)
        legal_moves.sort(key=lambda x: x[4], reverse=maxingPlayer)
        return [move[:-1] for move in legal_moves]
    
    
    def alphabeta(self, board: ChessBoard, depth: int, alpha: int, beta: int, maximizingPlayer: bool) -> tuple:
        """
        Alpha-Beta剪枝算法的实现。
        """
        if board.winner:
            if board.winner == self.team:
                return (10000000, None)
            else:
                return (-10000000, None)
        
        map_hash = board.map_hash()
        transposition_node = self.get_transposition_node(maximizingPlayer, map_hash)
        if transposition_node and transposition_node.depth >= depth and 60 - board.capture_count > depth+3 and not board.about_to_repeat():
            return (transposition_node.score, transposition_node.move)
            
        # print("depth: ", depth)
        pygame.event.get()
        team = self.team if maximizingPlayer else self.opponent
        if depth == 0:
            return (self.evaluate_board(board), None)
        if maximizingPlayer:
            maxEval = float('-inf')
            best_move = None
            moves = self.get_sorted_legal_moves(board, team)
            len_moves = len(moves)
            # print([move[4] for move in moves])
            for idx, move in enumerate(moves):
                new_board = board.copy_smallest()
                new_board.move_chess2(move[0], move[1], move[2], move[3])
                if idx < 25:
                    new_depth = depth - 1
                elif idx < 35:
                    new_depth = depth - 2
                else:
                    new_depth = depth - 3
                new_depth = max(new_depth, 0)
                # print(new_depth, depth, len_moves, idx)
                eval = self.alphabeta(new_board, new_depth, alpha, beta, False)[0]
                if eval > maxEval:
                    maxEval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    self.history_table[board.chessboard_map[move[0]][move[1]].team + "_" + board.chessboard_map[move[0]][move[1]].name][move[0]][move[1]][move[2]][move[3]] += 2 ** depth
                    break
            self.set_transposition_node(maximizingPlayer, TranspositionNode(len(board.get_chess()), depth, maxEval, best_move), map_hash)
            return (maxEval, best_move)
        else:
            minEval = float('inf')
            best_move = None
            moves = self.get_sorted_legal_moves(board, team)
            len_moves = len(moves)
            for idx, move in enumerate(moves):
                new_board = board.copy_smallest()
                new_board.move_chess2(move[0], move[1], move[2], move[3])
                if idx < 25:
                    new_depth = depth - 1
                elif idx < 35:
                    new_depth = depth - 2
                else:
                    new_depth = depth - 3
                new_depth = max(new_depth, 0)
                eval = self.alphabeta(new_board, new_depth, alpha, beta, True)[0]
                if eval < minEval:
                    minEval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.set_transposition_node(maximizingPlayer, TranspositionNode(len(board.get_chess()), depth, minEval, best_move), map_hash)
            return (minEval, best_move)
    
    def get_next_step(self, chessboard: ChessBoard):
        max_val = -100000000
        next_step = (0, 0, 0, 0)
        num_legal_moves = len(self.get_legal_moves(chessboard, self.team))
        self.delete_old_transposition_nodes(len(chessboard.get_chess()))
        depth = 5
        start_time = time.time()
        new_board = chessboard.copy_smallest()
        score, next_step = self.alphabeta(new_board, depth, float('-inf'), float('inf'), True)
        end_time = time.time()
        print("score:", score, "\tlen(transposition_table_max):", len(self.transposition_table_max), "\ttime:", round(end_time - start_time, 1))
        return next_step
