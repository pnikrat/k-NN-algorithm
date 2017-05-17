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
        print(self.training_data[1])
        self.choose_k_value()
        self.choose_distance_calculation_method()
        classification_flow = ClassificationFlow(self)
        classification_flow.run_user_classification_loop()

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
                    self.training_data.append((np.array(row[:-1], dtype='float64'), row[-1]))

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

    @staticmethod
    def is_numeric(value):
        try:
            value = float(value)
            return True
        except ValueError:
            if value == 'q':
                raise UserQuitException("Quitting...")
            else:
                return False

class EuclidesDistance:
    @staticmethod
    def calculate_distance(self, caseA, caseB):
        difference_vector = (caseA - caseB)**2
        #squared_difference_vector = difference_vector**2
        return np.sqrt(difference_vector.sum())


class ManhattanDistance:
    @staticmethod
    def calculate_distance(self, caseA, caseB):
        pass


class ClassificationFlow:
    def __init__(self, single_algorithm_case):
        self.studied_case = single_algorithm_case
        self.user_provided_test_case = None
        self.user_provided_test_attribute = None
        self.current_attribute_label = None

    def run_user_classification_loop(self):
        print("Press q to quit during classification")
        while True:
            try:
                self.user_provided_test_case = self.provide_test_case()
                print("Classifying...")
                print(self.user_provided_test_case)
                self.classify()
            except UserQuitException as quitmessage:
                print(quitmessage.args[0])
                return

    def provide_test_case(self):
        single_test_case = []
        for index, attribute_label in enumerate(self.studied_case.training_data[0]):
            if index == len(self.studied_case.training_data[0]) - 1:
                break
            self.current_attribute_label = attribute_label
            self.provide_test_attribute()
            single_test_case.append(self.user_provided_test_attribute)
        return np.array(single_test_case)

    def provide_test_attribute(self):
        chosen_test_attribute = input("Input numeric value for attribute named: " + self.current_attribute_label)
        if UserInputEvaluation.is_numeric(chosen_test_attribute):
            print("Chosen value " + chosen_test_attribute + " for attribute " + self.current_attribute_label)
            self.user_provided_test_attribute = chosen_test_attribute
            return
        else:
            UserInputEvaluation.retry_user_input(self.provide_test_attribute, "Test attribute must be numeric")

    def classify(self):
        pass

class UserQuitException(Exception):
    pass