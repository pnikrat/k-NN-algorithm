import csv
import numpy as np


class SingleAlgorithmCase:
    def __init__(self, cli_argument):
        self.file_handler = CsvFileHandler(cli_argument)
        self.training_data = []
        self.k_value = None

    def start(self):
        self.file_handler.import_training_data()
        self.training_data = self.file_handler.get_training_data_with_labels()
        self.choose_k_value()

    def choose_k_value(self):
        k_value = input("Please provide k value for algorithm: ")
        if UserInputEvaluation.is_k_value_valid(k_value):
            print("Chosen k value = " + k_value)
            self.k_value = int(k_value)
        else:
            UserInputEvaluation.retry_user_input(self.choose_k_value, "k value must be integer and positive")


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


class UserInputEvaluation:
    @staticmethod
    def is_k_value_valid(value):
        try:
            value = int(value)
            if value > 0:
                return True
            else:
                return False
        except ValueError:
            return False

    @staticmethod
    def retry_user_input(prompt_function, why_retry_message):
        print(why_retry_message)
        prompt_function()