from __future__ import annotations
from typing import Optional, Dict
import enum

import enums
import numpy as np
import pygame
from environment import track


class View:
    def __init__(self, racetrack_: track.RaceTrack):
        self.track: np.ndarray = racetrack_.track

        self.screen_width: int = 1500
        self.screen_height: int = 1000
        self.cell_pixels: int = 10
        self.screen: Optional[pygame.Surface] = None
        self.background: Optional[pygame.Surface] = None
        self.track_surface: Optional[pygame.Surface] = None

        self.background_color: pygame.Color = pygame.Color('grey10')
        self.color_lookup: Dict[enums.Square, pygame.Color] = {}

        self.user_event: UserEvent = UserEvent.NONE

        self.build_color_lookup()
        self.load_racetrack()

    @property
    def screen_size(self) -> tuple:
        return self.screen_width, self.screen_height

    # noinspection SpellCheckingInspection
    def build_color_lookup(self):
        self.color_lookup = {
            enums.Square.TRACK: pygame.Color('darkgrey'),
            enums.Square.GRASS: pygame.Color('forestgreen'),
            enums.Square.START: pygame.Color('yellow2'),
            enums.Square.END: pygame.Color('goldenrod2'),
            enums.Square.CAR: pygame.Color('deepskyblue2')
        }

    def load_racetrack(self):
        self.set_sizes()
        for index, track_value in np.ndenumerate(self.track):
            row, col = index
            square = enums.Square(track_value)
            self.draw_square(row, col, square, self.track_surface)
        self.background.blit(source=self.track_surface, dest=(0, 0))

    def set_sizes(self):
        # size window for track and set cell_pixels
        rows, cols = self.track.shape
        self.cell_pixels = int(min(self.screen_height / rows, self.screen_width / cols))
        self.screen_width = cols * self.cell_pixels
        self.screen_height = rows * self.cell_pixels

        self.background = pygame.Surface(size=self.screen_size)
        # self.background.convert()
        self.background.fill(self.background_color)
        self.track_surface = pygame.Surface(size=self.screen_size)
        # self.track_surface.convert()
        self.track_surface.fill(self.background_color)

    def draw_square(self, row: int, col: int, square: enums.Square, surface: pygame.Surface):
        color: pygame.Color = self.color_lookup[square]
        left: int = col * self.cell_pixels
        top: int = row * self.cell_pixels
        width: int = self.cell_pixels - 1
        height: int = self.cell_pixels - 1

        # doesn't like named parameters
        rect: pygame.Rect = pygame.Rect(left, top, width, height)
        pygame.draw.rect(surface, color, rect)

    def open_window(self):
        self.screen = pygame.display.set_mode(size=self.screen_size)
        pygame.display.set_caption('Racetrack finite MDP control Q-learning')
        # self.background = pygame.Surface(size=self.screen_size).convert()
        self.background = self.background.convert()
        self.track_surface = self.track_surface.convert()
        pygame.key.set_repeat(500, 50)
        # self.background.fill(self.background_color)

    def display_and_wait(self):
        while self.user_event != UserEvent.QUIT:
            self.update_screen()
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_SPACE]:
            #     self.user_event = UserEvent.SPACE
            # else:
            self.wait_for_event_of_interest()
            self.handle_event()

    def update_screen(self):
        self.screen.blit(source=self.background, dest=(0, 0))
        pygame.display.flip()

    def wait_for_event_of_interest(self):
        self.user_event = UserEvent.NONE
        while self.user_event == UserEvent.NONE:
            # replaced: for event in pygame.event.get():
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self.user_event = UserEvent.QUIT
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.user_event = UserEvent.SPACE
            # else:
            #     keys = pygame.key.get_pressed()
            #     if keys[pygame.K_SPACE]:
            #         self.user_event = UserEvent.SPACE
            #         print("Space")

            # elif event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         self.user_event = UserEvent.SPACE
            #     else:
            #         self.user_event = UserEvent.NONE
            # elif event.type == pygame.KEYUP:
            #     print("up")
            #     self.user_event = UserEvent.NONE
            # # else:
            # #     self.user_event = UserEvent.NONE
            #
            # if self.user_event != UserEvent.NONE:
            #     break
            # elif event.type == pygame.KEYUP:
            #     print("up")

    def handle_event(self):
        if self.user_event == UserEvent.QUIT:
            self.close_window()
            # sys.exit()
        elif self.user_event == UserEvent.SPACE:
            self.draw_car()

    def draw_car(self):
        rng: np.random.Generator = np.random.default_rng()
        row = rng.choice(self.track.shape[0])
        col = rng.choice(self.track.shape[1])

        # print(self.track.flatten())
        # flat_index = rng.choice(self.track.flatten())
        # print(flat_index)
        # row, col = np.unravel_index(flat_index, self.track.shape)
        # print(row, col)
        self.draw_square(row, col, enums.Square.CAR, self.background)

    def close_window(self):
        # pygame.display.quit()
        pygame.quit()


class UserEvent(enum.IntEnum):
    NONE = 0
    QUIT = 1
    SPACE = 2

