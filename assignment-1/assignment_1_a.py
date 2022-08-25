from csvloader import CSVLoader
from linearregressionmodel import LinearRegressionModel

FILE_NAME = 'data/length_weight.csv'
values = CSVLoader.load(FILE_NAME)

model = LinearRegressionModel()
