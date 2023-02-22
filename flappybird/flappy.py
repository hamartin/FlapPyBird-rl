'''This file represents the game itself.'''


import os
import random
import sys

import pygame
import pygame.locals as pl

from .pipes import Pipes
from .player import Player
from .score import Score


class Flappy:

    '''This class represents the game itself.'''

    def __init__(self, args):
        # Arguments given on the command line
        self.args = args

        # Defaults
        self.fps = 30
        self.screenHeight = 512
        self.screenWidth = 288

        # Empty defaults
        self.movementInfo = {}
        self.sprites = {}
        self.spritesPaths = {}

        self._initPygame()
        self._initSprites()

        self.pipes = Pipes(self.screen)
        self.player = Player(self.screen, self.fps, self.args)
        self.score = Score(self.screen)

    def _initPygame(self):
        pygame.init()
        self.fpsClock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.screenWidth,
                                               self.screenHeight))
        pygame.display.set_caption('Flappy Bird')

    def _initSprites(self):
        currentDir = os.path.dirname(os.path.abspath(__file__))
        spritesDir = os.path.join(currentDir, 'assets', 'sprites')

        self.spritesPaths['background'] = (
                os.path.join(spritesDir, 'background-day.png'),
                os.path.join(spritesDir, 'background-night.png')
                )
        self.setRandomBackground()

    def checkForEvents(self):
        for event in pygame.event.get():
            if event.type == pl.QUIT or (event.type == pl.KEYDOWN and event.key == pl.K_ESCAPE):
                pygame.quit()
                sys.exit()

    def run(self):
        while True:
            self.setRandomBackground()
            self.player.setRandomPlayerSprites()

            self.movementInfo = self.showWelcomeAnimation()

    def setRandomBackground(self):
        index = random.randint(0, len(self.spritesPaths['background'])-1)
        self.sprites['background'] = pygame.image.load(
                self.spritesPaths['background'][index]).convert()

    def showWelcomeAnimation(self):
        playerXPos = int(self.screenWidth*0.2)
        height, _ = self.player.getHeightAndWidth(0)
        playerYPos = int((self.screenHeight-height)/2)
        self.player.setPos(playerXPos, playerYPos)

        while True:
            self.checkForEvents()
            self.player.updateAnimation()

            self.screen.blit(self.sprites['background'], (0, 0))
            self.player.blit()

            pygame.display.update()
            self.fpsClock.tick(self.fps)
