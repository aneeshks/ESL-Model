from pandas import read_csv

def read_data(filename):
    df = read_csv(filename, sep='\t')
    return df

def digit_float(f, digit=3):
    template = '{:.' + str(digit) + 'f}'
    return float(template.format(f))