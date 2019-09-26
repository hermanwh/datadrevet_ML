import pandas as pd
import sys

def main(filename):
    df = pd.read_csv(filename)
    print(df)
    print("Program completed")

if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)