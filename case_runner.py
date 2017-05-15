import csv
import numpy as np


class CsvFileHandler:
    training_labels = []
    training_data = []

    def __init__(self, filename):
        self.filename = filename

    def import_training_data(self):
        with open(self.filename, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for index, row in enumerate(csvreader):
                if index == 0:
                    self.training_labels = row
                else:
                    print(row)
                    self.training_data.append(np.array(row, dtype='int64'))


    def get_training_data_with_labels(self):
        return self.training_labels, self.training_data


class SingleAlgorithmCase:
    def __init__(self, cli_argument):
        self.file_handler = CsvFileHandler(cli_argument)
        self.file_handler.import_training_data()
        self.training_data = self.file_handler.get_training_data_with_labels()

    def start(self):
        print(self.training_data[0])
        print(self.training_data[1])
