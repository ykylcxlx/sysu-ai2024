import sys

from Game import *
from Dot import *
from ChessBoard import *
from ChessAI import ChessAI
from MyAI import *
import time
def main():
    USE_MYAI = True
    pygame.init() # 初始化pygame
    screen = pygame.display.set_mode((750, 667)) # 创建用来显示画面的对象（理解为相框）
    background_img = pygame.image.load("images/bg.jpg") # 游戏背景图片
    # chessboard_img = pygame.image.load("images/bg.png") # 游戏棋盘
    chessboard = ChessBoard(screen) # 创建棋盘对象
    clock = pygame.time.Clock() # 创建计时器
    game = Game(screen, chessboard) # 创建游戏对象（像当前走棋方、游戏是否结束等都封装到这个对象中）
    game.back_button.add_history(chessboard.get_chessboard_str_map())
    ai = ChessAI(game.computer_team) # 创建AI对象
    myai = MyAI(game.user_team)

    # 主循环
    while True:
        # pygame.time.delay(150) 
        # 显示游戏背景
        screen.blit(background_img, (0, 0))
        screen.blit(background_img, (0, 270))
        screen.blit(background_img, (0, 540))

        # # 显示棋盘
        # # screen.blit(chessboard_img, (50, 50))
        # chessboard.show()
        #
        # # 显示棋盘上的所有棋子
        # # for line_chess in chessboard_map:
        # for line_chess in chessboard.chessboard_map:
        #     for chess in line_chess:
        #         if chess:
        #             # screen.blit(chess[0], chess[1])
        #             chess.show()
        
        chessboard.show_chessboard_and_chess() # 显示棋盘以及棋子
        ClickBox.show() # 标记点击的棋子
        Dot.show_all() # 显示可以落子的位置图片
        game.show() # 显示游戏相关信息
        pygame.event.get()
        pygame.display.update() # 显示screen这个相框的内容（此时在这个相框中的内容像照片、文字等会显示出来）
        pygame.event.get()

        
        # AI行动
        if not game.show_win and not game.show_draw and game.AI_mode and game.get_player() == ai.team:
            start_time = time.time()
            if game.back_button.is_repeated():
                print("获胜...")
                game.set_win(game.get_player())
            else:
                # AI预测下一步
                cur_row, cur_col, nxt_row, nxt_col = ai.get_next_step(chessboard)
                # 选择棋子
                ClickBox(screen, cur_row, cur_col)
                # 下棋子
                chessboard.move_chess(nxt_row, nxt_col)
                # 清理「点击对象」
                ClickBox.clean()
                # 检测落子后，是否产生了"将军"功能
                if chessboard.judge_attack_general(game.get_player()):
                    print("将军....")
                    # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
                    if chessboard.judge_win(game.get_player()):
                        print("获胜...")
                        game.set_win(game.get_player())
                    else:
                        # 如果攻击到对方，则标记显示"将军"效果
                        game.set_attack(True)
                else:
                    if chessboard.judge_win(game.get_player()):
                        print("获胜...")
                        game.set_win(game.get_player())
                    game.set_attack(False)    
                
                if chessboard.judge_draw():
                    print("和棋...")
                    game.set_draw()

                # 落子之后，交换走棋方
                game.back_button.add_history(chessboard.get_chessboard_str_map())
                game.exchange()
            end_time = time.time()
            print("opponent move time: ", round(end_time - start_time, 1))
        else:
            if not USE_MYAI:
                # 事件检测（例如点击了键盘、鼠标等）
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()  # 退出程序
                    if game.back_button.is_repeated():
                        print("获胜...")
                        game.set_win(game.get_player())
                    else:
                        # 如果游戏没有获胜方，则游戏继续，否则一直显示"获胜"
                        if not game.show_win and not game.show_draw:
                            # 检测是否点击了"可落子"对象
                            clicked_dot = Dot.click()
                            if clicked_dot:
                                chessboard.move_chess(clicked_dot.row, clicked_dot.col)
                                # 清理「点击对象」、「可落子位置对象」
                                Dot.clean_last_position()
                                ClickBox.clean()
                                # 检测落子后，是否产生了"将军"功能
                                if chessboard.judge_attack_general(game.get_player()):
                                    print("将军....")
                                    # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
                                    if chessboard.judge_win(game.get_player()):
                                        print("获胜...")
                                        game.set_win(game.get_player())
                                    else:
                                        # 如果攻击到对方，则标记显示"将军"效果
                                        game.set_attack(True)
                                else:
                                    if chessboard.judge_win(game.get_player()):
                                        print("获胜...")
                                        game.set_win(game.get_player())
                                    game.set_attack(False)
                                
                                if chessboard.judge_draw():
                                    print("和棋...")
                                    game.set_draw()
                                
                                game.back_button.add_history(chessboard.get_chessboard_str_map())
                                # 落子之后，交换走棋方
                                game.exchange()
                                # 退出for，以便不让本次的鼠标点击串联到点击棋子
                                break

                            # 检查是否点击了棋子
                            # clicked_chess = Chess.get_clicked_chess(chessboard)
                            clicked_chess = Chess.get_clicked_chess(game.get_player(), chessboard)
                            if clicked_chess:
                                # 创建选中棋子对象
                                ClickBox(screen, clicked_chess.row, clicked_chess.col)
                                # 清除之前的所有的可以落子对象
                                Dot.clean_last_position()
                                # 计算当前被点击的棋子可以落子的位置
                                put_down_chess_pos = chessboard.get_put_down_position(clicked_chess)
                                # 根据当前被点击的棋子创建可以落子的对象
                                Dot.create_nums_dot(screen, put_down_chess_pos)

                            if game.back_button.clicked_back(chessboard, event):
                                break
            else:
                if game.show_win or game.show_draw:
                    continue
                # pygame.event.get()
                # AI预测下一步
                cur_row, cur_col, nxt_row, nxt_col = myai.get_next_step(chessboard)
                # 选择棋子
                ClickBox(screen, cur_row, cur_col)
                # 下棋子
                chessboard.move_chess(nxt_row, nxt_col)
                # 清理「点击对象」
                ClickBox.clean()
                # 检测落子后，是否产生了"将军"功能
                if chessboard.judge_attack_general(game.get_player()):
                    print("将军....")
                    # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
                    if chessboard.judge_win(game.get_player()):
                        print("获胜...")
                        game.set_win(game.get_player())
                    else:
                        # 如果攻击到对方，则标记显示"将军"效果
                        game.set_attack(True)
                else:
                    if chessboard.judge_win(game.get_player()):
                        print("获胜...")
                        game.set_win(game.get_player())
                    game.set_attack(False)    
                
                if chessboard.judge_draw():
                    print("和棋...")
                    game.set_draw()

                # 落子之后，交换走棋方
                game.back_button.add_history(chessboard.get_chessboard_str_map())
                game.exchange()



if __name__ == '__main__':
    main()
