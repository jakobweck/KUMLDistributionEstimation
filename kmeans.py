import numpy as np
import pandas as pd
import sys
import os


def readColumnsFromCSV(filepath, col1, col2):
    print (os.getcwd()+ "\\" + filepath)
    csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
    csvFrame = csvFrame.iloc[:,[10, 11]]
    print(csvFrame)

readColumnsFromCSV(sys.argv[1], 11, 12)