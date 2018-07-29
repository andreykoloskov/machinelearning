import pandas
import sys

features = pandas.read_csv(sys.argv[1], index_col='match_id')

print(features.head())
