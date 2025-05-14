import os
import sys


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


import mathtools



if __name__ == "__main__":

    print(mathtools.factorial(5))
    print(mathtools.bernstein(0.5, 4, 2))
