'''This file represents utility functions and classes.'''


def shm(vals):
    '''Oscilates the value of vals['val'] between 8 and -8'''
    if abs(vals['val']) == 8:
        vals['dir'] *= -1
    if vals['dir'] == 1:
        vals['val'] += 1
    else:
        vals['val'] -= 1
