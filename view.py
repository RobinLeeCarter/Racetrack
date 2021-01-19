from __future__ import annotations
from typing import Optional, Dict

import numpy as np
import pygame

import enums
from environment import track
import environment
import agent


class View:
    def __init__(self, racetrack_: track.RaceTrack):
        self.racetrack: track.RaceTrack = racetrack_

        self.screen_width: int = 1500
        self.screen_height: int = 1000
        self.cell_pixels: int = 10
        self.screen: Optional[pygame.Surface] = None
        self.background: Optional[pygame.Surface] = None
        self.track_surface: Optional[pygame.Surface] = None

        self.background_color: pygame.Color = pygame.Color('grey10')
        self.color_lookup: Dict[enums.Square, pygame.Color] = {}

        self.user_event: enums.UserEvent = enums.UserEvent.NONE

        self.t: int = 0
        self.episode: Optional[agent.Episode] = None

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
        self.track_surface.fill(self.background_color)
        for index, track_value in np.ndenumerate(self.racetrack.track):
            row, col = index
            square = enums.Square(track_value)
            self.draw_square(row, col, square, self.track_surface)
        self.copy_track_into_background()

    def copy_track_into_background(self):
        self.background.blit(source=self.track_surface, dest=(0, 0))

    def set_sizes(self):
        # size window for track and set cell_pixels
        rows, cols = self.racetrack.track.shape
        self.cell_pixels = int(min(self.screen_height / rows, self.screen_width / cols))
        self.screen_width = cols * self.cell_pixels
        self.screen_height = rows * self.cell_pixels

        self.background = pygame.Surface(size=self.screen_size)
        self.track_surface = pygame.Surface(size=self.screen_size)

    def draw_square(self, row: int, col: int, square: enums.Square, surface: pygame.Surface) -> pygame.Rect:
        color: pygame.Color = self.color_lookup[square]
        left: int = col * self.cell_pixels
        top: int = row * self.cell_pixels
        width: int = self.cell_pixels - 1
        height: int = self.cell_pixels - 1

        # doesn't like named parameters
        rect: pygame.Rect = pygame.Rect(left, top, width, height)
        pygame.draw.rect(surface, color, rect)
        return rect

    def open_window(self):
        self.screen = pygame.display.set_mode(size=self.screen_size)
        pygame.display.set_caption('Racetrack finite MDP control Q-learning')
        # self.background = pygame.Surface(size=self.screen_size).convert()
        self.background = self.background.convert()
        self.track_surface = self.track_surface.convert()
        pygame.key.set_repeat(500, 50)
        # self.background.fill(self.background_color)

    def display_and_wait(self):
        while self.user_event != enums.UserEvent.QUIT:
            self.put_background_on_screen()
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_SPACE]:
            #     self.user_event = enums.enums.UserEvent.SPACE
            # else:
            self.wait_for_event_of_interest()
            self.handle_event()

    def display_episode(self, episode_: agent.Episode) -> enums.UserEvent:
        # print(episode_.trajectory)
        # print(f"len(self.episode.trajectory) = {len(episode_.trajectory)}")
        self.copy_track_into_background()
        self.put_background_on_screen()
        self.episode = episode_
        self.t = 0
        terminal = len(self.episode.trajectory) - 1  # terminal state
        while self.user_event != enums.UserEvent.QUIT and self.t <= terminal:
            # need to pass through for terminal state to display penultimate state
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_SPACE]:
            #     self.user_event = enums.enums.UserEvent.SPACE
            # else:
            self.wait_for_event_of_interest()
            self.handle_event()
        return self.user_event

    def put_background_on_screen(self):
        self.screen.blit(source=self.background, dest=(0, 0))
        pygame.display.flip()

    def wait_for_event_of_interest(self):
        self.user_event = enums.UserEvent.NONE
        while self.user_event == enums.UserEvent.NONE:
            # replaced: for event in pygame.event.get():
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self.user_event = enums.UserEvent.QUIT
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.user_event = enums.UserEvent.SPACE
            # else:
            #     keys = pygame.key.get_pressed()
            #     if keys[pygame.K_SPACE]:
            #         self.user_event = enums.UserEvent.SPACE
            #         print("Space")

            # elif event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         self.user_event = enums.UserEvent.SPACE
            #     else:
            #         self.user_event = enums.UserEvent.NONE
            # elif event.type == pygame.KEYUP:
            #     print("up")
            #     self.user_event = enums.UserEvent.NONE
            # # else:
            # #     self.user_event = enums.UserEvent.NONE
            #
            # if self.user_event != enums.UserEvent.NONE:
            #     break
            # elif event.type == pygame.KEYUP:
            #     print("up")

    def handle_event(self):
        if self.user_event == enums.UserEvent.QUIT:
            self.close_window()
            # sys.exit()
        elif self.user_event == enums.UserEvent.SPACE:
            state: environment.State = self.episode.trajectory[self.t].state
            if not state.is_terminal:
                self.draw_car_at_state(state)
            self.t += 1
            # self.draw_random_car()

    def draw_car_at_state(self, state: environment.State):
        row, col = self.racetrack.get_index(state.x, state.y)
        # print(f"t={self.t} x={state.x} y={state.y} row={row} col={col}")
        rect: pygame.Rect = self.draw_square(row, col, enums.Square.CAR, self.background)
        self.screen.blit(source=self.background, dest=rect, area=rect)
        pygame.display.update(rect)
        # self.screen.blit(source=self.background, dest=(0, 0))
        # pygame.display.flip()

    def draw_random_car(self):
        rng: np.random.Generator = np.random.default_rng()
        row = rng.choice(self.racetrack.track.shape[0])
        col = rng.choice(self.racetrack.track.shape[1])

        # print(self.track.flatten())
        # flat_index = rng.choice(self.track.flatten())
        # print(flat_index)
        # row, col = np.unravel_index(flat_index, self.track.shape)
        # print(row, col)
        self.draw_square(row, col, enums.Square.CAR, self.background)

    def close_window(self):
        # pygame.display.quit()
        pygame.quit()
