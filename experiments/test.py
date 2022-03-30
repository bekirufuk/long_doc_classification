import os, glob
import pandas as pd


files_list = glob.glob('data/patentsview/chunks/*.csv')
s = 0
for file in files_list:
    df = pd.read_csv(file)
    print()