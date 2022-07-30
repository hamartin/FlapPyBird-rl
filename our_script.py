import flappy as flp
import itertools
import random
import pygame
import torch


def run():
    screen, movementInfo = flp.main()
    for episode in itertools.count():
        print('episode', episode)
        # x3 = pygame.surfarray.pixels3d(screen)
        # print('x3', x3)
        flappy = flp.Flappy(movementInfo)
        end_game = False
        screenbuf, end_game, res = flappy.step(False)
        screenbuf_t = torch.from_numpy(screenbuf)
        for step_num in itertools.count():
            # print('screenbuf_t.size()', screenbuf_t.size(), screenbuf_t.dtype)
            action = random.random() > 0.9
            screenbuf, end_game, res = flappy.step(action)
            if end_game:
                reward = res['score']
                break
            screenbuf_t = torch.from_numpy(screenbuf)
        print('episode reward', reward)

if __name__ == '__main__':
    run()
