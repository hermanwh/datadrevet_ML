import pandas as pd
import sys

def main(file, column, target_file):
    print("Loading file {}".format(file))
    df_iris = pd.read_csv(file).drop(column, 1)
    print("Writing file {}".format(target_file))
    df_iris.to_csv(target_file, index=False)
    print("Program completed")

# usage: python drop_column.py file column targetfile
if __name__ == "__main__":
    file = sys.argv[1]
    column = sys.argv[2]
    target_file = sys.argv[3]
    
    main(file, column, target_file)