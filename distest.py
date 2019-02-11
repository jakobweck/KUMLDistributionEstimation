import numpy as np
import pandas as pd
import sys
import os


def readcsv(filepath):
	if os.name == 'nt':
		print (os.getcwd()+ "\\" + filepath)
		csvFrame = pd.read_csv(os.getcwd()+ "\\" + filepath)
	else:
    		csvFrame = pd.read_csv(filepath)
	print(csvFrame)

if __name__ == "__main__":
	readcsv(sys.argv[1])
