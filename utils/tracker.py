"""Custom Log Object - Tracker

This module open, append and close a log file.
"""
import datetime


class Tracker:
    def __init__(self, name='track.log'):
        '''Initialize the tracker, open file ```name```'''
        self.name = name
        self.file = open(self.name, 'w')
        self.initialized = False

    def __call__(self, _in):
        '''Append to the self.file'''
        if not self.initialized:
            # in case of first opening
            self.file.write('Tracker Initialized - ' + str(
                datetime.date.today()) + '\n')
            self.initialized = not self.initialized

        self.file.write(_in + '\n')

    def _close(self):
        '''Close the self.file'''
        self.file.close()
