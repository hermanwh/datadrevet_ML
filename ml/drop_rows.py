import pandas as pd
import sys

def main(file, drop, target_file):
    print("Loading file {}".format(file))
    df = pd.read_csv(file)
    df = df.iloc[drop:, :]
    print("Writing file {}".format(target_file))
    df.to_csv(target_file, index=False)
    print("Program completed")

# usage: python drop_column.py file 100 targetfile
if __name__ == "__main__":
    file = sys.argv[1]
    drop = int(sys.argv[2])
    target_file = sys.argv[3]
    
    main(file, drop, target_file)