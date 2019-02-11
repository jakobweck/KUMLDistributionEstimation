import numpy as np
import pandas as pd
import sys
import os



def readcsv(filepath):
    print (os.getcwd()+ "\\" + filepath)
    csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
    print(csvFrame)

readcsv(sys.argv[1])
