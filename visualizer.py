import pygame
import numpy as np


class Visualizer:

    def __init__(self, data):
        self.data = data
        self.window = False

    # HELPFUL FUNCTIONS

    def setup(self):
        '''
        Does things which occur only at the beginning
        '''
        self.cell_edge = 9
        self.width = 100 * self.cell_edge
        self.height = 100 * self.cell_edge
        self.create_window()
        self.window = True

    def create_window(self):
        '''
        Creates window and assigns self.display variable
        '''
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racetrack")

    def close_window(self):
        self.window = False
        pygame.quit()

    def draw(self, state=np.array([])):
        self.display.fill(0)
        for i in range(100):
            for j in range(100):
                # if self.data.racetrack[i, j] != -1:
                #     if self.data.racetrack[i, j] == 0:
                #         color = (255, 0, 0)
                #     elif self.data.racetrack[i, j] == 1:
                #         color = (255, 255, 0)
                #     elif self.data.racetrack[i, j] == 2:
                #         color = (0, 255, 0)
                #     pygame.draw.rect(self.display, color,
                #                      ((j * self.cell_edge, i * self.cell_edge), (self.cell_edge, self.cell_edge)), 1)
                if self.data[i, j] != -1:
                    if self.data[i, j] == 0:
                        color = (255, 0, 0)
                    elif self.data[i, j] == 1:
                        color = (255, 255, 0)
                    elif self.data[i, j] == 2:
                        color = (0, 255, 0)
                    rect = pygame.Rect(j * self.cell_edge, i * self.cell_edge, self.cell_edge, self.cell_edge)
                    pygame.draw.rect(self.display,
                                     color,
                                     rect,
                                     1)

        if len(state) > 0:
            rect = pygame.Rect(state[1] * self.cell_edge, state[0] * self.cell_edge, self.cell_edge, self.cell_edge)
            pygame.draw.rect(self.display,
                             (0, 0, 255),
                             rect,
                             1)

        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.loop = False
                self.close_window()
                return 'stop'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.loop = False

        return None

    def visualize_racetrack(self, state=np.array([])):
        '''
        Draws Racetrack in a pygame window
        '''
        if self.window == False:
            self.setup()
        self.loop = True
        while (self.loop):
            ret = self.draw(state)
            if ret != None:
                return ret
