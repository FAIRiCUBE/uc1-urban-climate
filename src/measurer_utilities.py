import os
import pandas as pd
def bytes_to(bytes_value, to, bsize=1024):
    # convert bytes to megabytes, etc.
    a = {'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5, 'e': 6}
    r = float(bytes_value)
    for i in range(a[to]):
        r = r / bsize
    return r

def compare_benchmarks(benchmark_folder = "benchmarks"):
    # read all csv files in the folder
    files = [f for f in os.listdir(benchmark_folder) if f.endswith('.csv')]
    # read in pandas
    dfs = [pd.read_csv(os.path.join(benchmark_folder, f), index_col=0) for f in files]
    # merge all dataframes by index, index is the first column
    df = pd.merge(dfs[0], dfs[1], on='Measure')
    return df
