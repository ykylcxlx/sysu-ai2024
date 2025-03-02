import pygame


class Chess(pygame.sprite.Sprite):
    """
    棋子类
    """

    def __init__(self, screen, chess_name, row, col, smallest=False):
        if smallest:
            self.smallest = True
            return
        super().__init__()
        self.screen = screen
        self.smallest = False
        self.team = chess_name[0]  # 队伍（红方 r、黑方b）
        self.name = chess_name[2]  # 名字（炮p、马m等）
        self.image = pygame.image.load("images/" + chess_name + ".png")
        self.top_left = (50 + col * 57, 50 + row * 57)
        self.rect = self.image.get_rect()
        self.rect.topleft = (50 + col * 57, 50 + row * 57)
        self.row, self.col = row, col
    
    def copy_smallest(self):
        new_chess = Chess(None, None, None, None, True)
        new_chess.team = self.team
        new_chess.name = self.name
        new_chess.row = self.row
        new_chess.col = self.col
        return new_chess

    def show(self):
        # self.screen.blit(self.image, self.top_left)
        self.screen.blit(self.image, self.rect)

    @staticmethod
    # def get_clicked_chess(chessboard):
    def get_clicked_chess(player, chessboard):
        """
        获取被点击的棋子
        """
        for chess in chessboard.get_chess():
            if pygame.mouse.get_pressed()[0] and chess.rect.collidepoint(pygame.mouse.get_pos()):
                if player == chess.team:
                    print(chess.name + "被点击了")
                    return chess

    def update_position(self, new_row, new_col):
        """
        更新要显示的图片的坐标
        """
        self.row = new_row
        self.col = new_col
        if not self.smallest:
            self.rect.topleft = (50 + new_col * 57, 50 + new_row * 57)
