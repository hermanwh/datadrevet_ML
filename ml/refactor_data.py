import pandas as pd
import sys

def drop_columns(df, cols):
    for col in cols:
        df = df.drop(col,1)
    return df

def main(filename, targetfilename, remove_rows, remove_columns): 
    ext = filename[-4:]
    if ext == '.csv':
        print("Loading file {}".format(filename))
        df = pd.read_csv(filename)
        df = drop_columns(df, remove_columns)
        df = df.iloc[remove_rows:, :]
        print(df)
        print("Writing file {}".format(targetfilename))
        df.to_csv(targetfilename, index=False)
    elif ext == '.xls':
        print("Loading file {}".format(filename))
        df = pd.read_excel(filename)
        df = drop_columns(df, remove_columns)
        df = df.iloc[remove_rows:, :]
        print(df)
        print("Writing file {}".format(targetfilename))
        df.to_csv(targetfilename, index=False)
    else:
        print("Could not load data from file")
        print("Use .csv or .xls format")
    print("Program completed")

# usage: python print_data.py iris.csv iris_mod.csv 100 col1 col2 col3 ...
#        python print_data.py iris.xls iris_mod.csv 500 col1 col2 col3 ...
#        python print_data.py iris.csv iris_mod.csv 0
if __name__ == "__main__":
    filename = sys.argv[1]
    targetfilename = sys.argv[2]
    remove_rows = int(sys.argv[3])
    remove_columns = sys.argv[4:]
    main(filename, targetfilename, remove_rows, remove_columns)