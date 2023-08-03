'''
Sets the paths such that our notebooks can find the reduced_models modules.
'''

import os 
import sys

path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(path)