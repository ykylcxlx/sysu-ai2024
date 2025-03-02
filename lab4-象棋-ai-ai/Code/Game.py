import pygame
from ChessBoard import *
def opposite_player(player):
        return "r" if player == "b" else "b"

class BackChess(pygame.sprite.Sprite):
    def __init__(self, screen):
        self.screen = screen
        self.image = pygame.image.load("images/back.png")
        self.rect = self.image.get_rect()
        self.rect.topleft = (615, 280)
        self.history_map = list()

    def show(self):
        self.screen.blit(self.image, self.rect)

    def clicked_back(self, chessboard: ChessBoard, event):
        if event.type == pygame.MOUSEBUTTONUP and self.rect.collidepoint(event.pos):
            if len(self.history_map) <= 2:
                return False
            self.pop_history()
            self.pop_history()
            res = self.pop_history()
            chessboard.set_chessboard_str_map(res)
            self.add_history(res)
            return True

    def add_history(self, str_map):
        self.history_map.append(str_map)

    def pop_history(self):
        return self.history_map.pop()

    def is_repeated(self): # TODO: 连续重复走三步，则判断平局
        repeat = True
        history_len = len(self.history_map)
        if history_len < 10:
            repeat = False
        else:
            for i in range(1, 10, 4):
                if self.history_map[history_len - i] != self.history_map[history_len - i -4]:
                    repeat = False
                    break
            for i in range(3, 8, 4):
                if self.history_map[history_len - i] != self.history_map[history_len - i -4]:
                    repeat = False
                    break
        return repeat        
class Reset(pygame.sprite.Sprite):
    def __init__(self,screen):
        self.screen=screen
        self.image=pygame.image.load("images/btn_lose.png")
        self.rect = self.image.get_rect()
        self.rect.topleft = (615, 150)

    def show(self):
        self.screen.blit(self.image, self.rect) 

    def clicked_back(self, chessboard: ChessBoard, event):
        if event.type == pygame.MOUSEBUTTONUP and self.rect.collidepoint(event.pos):
            print("我方认输，重新开始！")
            return True
class Game(object):
    """
    游戏类
    """

    def __init__(self, screen, chessboard):
        self.screen = screen
        self.player = "r"  # 默认走棋的为红方r
        self.player_tips_r_image = pygame.image.load("images/red.png")
        self.player_tips_r_image_topleft = (550, 500)
        self.player_tips_b_image = pygame.image.load("images/black.png")
        self.player_tips_b_image_topleft = (550, 100)
        self.show_attack = False
        self.attack_player = 'r'
        self.show_attack_count = 0
        self.show_attack_time = 100
        self.attack_img = pygame.image.load("images/pk.png")
        self.show_win = False
        self.win_img = pygame.image.load("images/win.png")
        self.win_player = None
        self.show_win_count = 0
        self.show_win_time = 300

        self.show_lost = False
        self.lost_img = pygame.image.load("images/lost2.png")
        self.lost_player = None
        self.show_lost_count = 0
        self.show_lost_time = 300
        self.show_draw = False
        self.draw_img = pygame.image.load("images/draw.png")
        self.show_draw_count = 0
        self.show_draw_time = 300
        self.chessboard = chessboard

        self.AI_mode = True
        self.user_team = 'r'
        self.computer_team = 'b'
        self.reset_time = 0
        self.score = 0
        self.back_button = BackChess(screen)

        self.touxiang_button=Reset(screen)
    def get_player(self):
        """
        获取当前走棋方
        """
        return self.player

    def exchange(self):
        """
        交换走棋方
        """
        self.player = "r" if self.player == "b" else "b"
        return self.get_player()

    def reset_game(self):
        """重置游戏"""
        # 所谓的重置游戏，就是将棋盘恢复到默认，走棋方默认的红方
        # 重建新的默认棋子
        self.chessboard.create_chess()
        self.back_button.history_map = list()
        self.back_button.add_history(self.chessboard.get_chessboard_str_map())
        # 设置走棋方为黑方，这样对战两轮的游戏中，第一次我方为红方，红方先走，下一次我方为红方，黑方先走
        self.player = 'b'
        if self.reset_time ==1:
            exit(0)
        self.reset_time = 1
        print("Total games 2/2 ")
    def show(self):
        # 如果一方获胜，那么显示"赢"
        # 通过计时，实现显示一会"将军"之后，就消失
        if self.show_win:
            self.show_win_count += 1
            if self.show_win_count == self.show_win_time:
                self.show_win_count = 0
                self.show_win = False
                #self.reset_game()  # 游戏玩过一局之后，重置游戏
        if self.show_lost:
            self.show_lost_count += 1
            if self.show_lost_count == self.show_lost_time:
                self.show_lost_count = 0
                # self.show_win = False

        if self.show_win:
            if self.win_player == "b":
                self.screen.blit(self.win_img, (550, 100))
                self.screen.blit(self.lost_img, (550, 450))
            else:
                self.screen.blit(self.win_img, (550, 450))
                self.screen.blit(self.lost_img, (550, 100))
            return    

        # TODO: 和棋
        if self.show_draw:
            self.show_draw_count += 1
            if self.show_draw_count == self.show_draw_time:
                self.show_draw_count = 0
                # self.show_draw = False
                # self.reset_game()  # 游戏玩过一局之后，重置游戏
        if self.show_draw:
            self.screen.blit(self.draw_img, (550, 275))
            return

        # 通过计时，实现显示一会"将军"之后，就消失
        if self.show_attack:
            self.show_attack_count += 1
            if self.show_attack_count == self.show_attack_time:
                self.show_attack_count = 0
                self.show_attack = False

        if self.player == "r":
            self.screen.blit(self.player_tips_r_image, self.player_tips_r_image_topleft)
        else:
            self.screen.blit(self.player_tips_b_image, self.player_tips_b_image_topleft)

        if self.attack_player == "r":
            # 显示"将军"效果
            if self.show_attack:
                self.screen.blit(self.attack_img, (230, 400))
        else:
            # 显示"将军"效果
            if self.show_attack:
                self.screen.blit(self.attack_img, (230, 100))

        self.back_button.show()
        self.touxiang_button.show()

    def set_attack(self, show_attack):
        """
        标记"将军"效果
        """
        self.show_attack = show_attack
        self.attack_player = self.player

    def set_win(self, win_player):
        """
        设置获胜方
        """
        self.show_win = True
        self.win_player = win_player
        if self.win_player == 'b':
            self.score += 3
            print("score=",self.score)
        else:
            print("score=",self.score)
    def set_lost(self, lost_player):
        """
        设置失败方
        """
        self.show_lost = True
        self.lost_player = opposite_player(lost_player)## 改为

    def set_draw(self): # TODO: 和棋
        self.score += 1
        print("score=",self.score)
        self.show_draw = True