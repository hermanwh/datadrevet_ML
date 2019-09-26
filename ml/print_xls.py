import pandas as pd
import sys

def main(filename):
    df = pd.read_excel(filename).drop('Time',1).drop('e',1).drop('f',1).drop('g',1).drop('h',1).drop('i',1).drop('j',1)
    df_mod = df.iloc[100:, :]
    print(df_mod)
    print("Program completed")

if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)