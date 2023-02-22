#!/usr/bin/env python3


import argparse

from flappybird import Flappy


def getArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--silent', default=False, required=False,
        action='store_true', help='Makes the game completely muted.')

    return parser.parse_args()


if __name__ == '__main__':
    args = getArguments()
    Flappy(args=args).run()
