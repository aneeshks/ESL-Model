from pandas import read_csv

def read_data(filename):
    df = read_csv(filename, sep='\t')
    return df
