import csv
import numpy as np


class SingleAlgorithmCase:
    def __init__(self, cli_argument):
        self.file_handler = CsvFileHandler(cli_argument)
        self.training_data = []
        self.k_value = None
        self.calculate_distance_function = None

    def start(self):
        self.file_handler.import_training_data()
        self.training_data = self.file_handler.get_training_data_with_labels()
        self.choose_k_value()
        self.choose_distance_calculation_method()
        print(self.calculate_distance_function)

    def choose_k_value(self):
        k_value = input("Please provide k value for algorithm: ")
        if UserInputEvaluation.is_k_value_valid(k_value):
            print("Chosen k value = " + k_value)
            self.k_value = int(k_value)
        else:
            UserInputEvaluation.retry_user_input(self.choose_k_value, "k value must be integer and positive")

    def choose_distance_calculation_method(self):
        choice_of_method = input("Classify using Euclides distance (E) or Manhattan distance (M):")
        if isinstance(choice_of_method, str):
            choice_of_method = choice_of_method.lower()
            if choice_of_method == 'e':
                self.calculate_distance_function = EuclidesDistance.calculate_distance
                return
            elif choice_of_method == 'm':
                self.calculate_distance_function = ManhattanDistance.calculate_distance
                return
        UserInputEvaluation.retry_user_input(self.choose_distance_calculation_method, "Wrong input. Type letter E or M")


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


class EuclidesDistance:
    @staticmethod
    def calculate_distance(self):
        pass


class ManhattanDistance:
    @staticmethod
    def calculate_distance(self):
        pass
