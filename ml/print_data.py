import pandas as pd
import sys

def drop_columns(df, cols):
    for col in cols:
        df = df.drop(col,1)
    return df

def main(filename, remove_rows, remove_columns): 
    ext = filename[-4:]
    if ext == '.csv':
        df = pd.read_csv(filename)
        df = drop_columns(df, remove_columns)
        df = df.iloc[remove_rows:, :]
        print(df)
    elif ext == '.xls':
        df = pd.read_excel(filename)
        df = drop_columns(df, remove_columns)
        df = df.iloc[remove_rows:, :]
        print(df)
    else:
        print("Could not load data from file")
        print("Use .csv or .xls format")
    print("Program completed")

# usage: python print_data.py iris_mod.csv 100 col1 col2 col3 ...
#        python print_data.py iris_mod.xls 500 col1 col2 col3 ...
if __name__ == "__main__":
    filename = sys.argv[1]
    remove_rows = int(sys.argv[2])
    remove_columns = sys.argv[3:]
    main(filename, remove_rows, remove_columns)