'''This file represents the player in the game.'''


import os
import random

import pygame

from itertools import cycle

from .utils import shm


class Player:

    '''This class represents the player in the game.'''

    def __init__(self, screen, fps, args):
        self.screen = screen
        self.fps = fps
        self.args = args

        # Empty defaults
        self.sprites = ()
        self.spritesPaths = ()

        self.yPos = 0
        self.xPos = 0

        # defaults
        self.index = 0                          # Points to the bird color of choice right now
        self.indexGen = cycle((0, 1, 2, 1))     # Cycles through the flap animation
        self.loopIter = 0                       # Used to change self.index every 5th iteration
        self.shmVals = {'val': 0, 'dir': 1}     # Used to animate the flapping}

        self._initSprites()

    def _initSprites(self):
        currentDir = os.path.dirname(os.path.abspath(__file__))
        spritesDir = os.path.join(currentDir, 'assets', 'sprites')

        self.spritesPaths = (
                (
                    os.path.join(spritesDir, 'redbird-upflap.png'),
                    os.path.join(spritesDir, 'redbird-midflap.png'),
                    os.path.join(spritesDir, 'redbird-downflap.png')
                    ),
                (
                    os.path.join(spritesDir, 'bluebird-upflap.png'),
                    os.path.join(spritesDir, 'bluebird-midflap.png'),
                    os.path.join(spritesDir, 'bluebird-downflap.png')
                    ),
                (
                    os.path.join(spritesDir, 'yellowbird-upflap.png'),
                    os.path.join(spritesDir, 'yellowbird-midflap.png'),
                    os.path.join(spritesDir, 'yellowbird-downflap.png')
                    )
                )
        self.setRandomPlayerSprites()

    def blit(self):
        self.screen.blit(self.sprites[self.index], (self.xPos, self.yPos+self.shmVals['val']))

    def getHeightAndWidth(self, index: int) -> tuple:
        assert index in (0, 1, 2)
        height = self.sprites[index].get_height()
        width = self.sprites[index].get_width()
        return (height, width)

    def getPos(self) -> tuple:
        return (self.xPos, self.yPos)

    def setPos(self, xPos: int, yPos: int) -> None:
        self.xPos = xPos
        self.yPos = yPos

    def setXPos(self, xPos: int) -> None:
        self.xPos = xPos

    def setYPos(self, yPos: int) -> None:
        self.yPos = yPos

    def setRandomPlayerSprites(self) -> None:
        index = random.randint(0, len(self.spritesPaths)-1)
        self.sprites = (
                pygame.image.load(self.spritesPaths[index][0]).convert_alpha(),
                pygame.image.load(self.spritesPaths[index][1]).convert_alpha(),
                pygame.image.load(self.spritesPaths[index][2]).convert_alpha()
                )

    def updateAnimation(self) -> None:
        if (self.loopIter+1) % 5 == 0:
            self.index = next(self.indexGen)
        self.loopIter = (self.loopIter+1) % self.fps
        shm(self.shmVals)
