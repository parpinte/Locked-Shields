from torch.utils.hipify.hipify_python import preprocessor


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None


    # function to loas the data for that you need to have pandas imported as pd
    def load_data(self):
        """ here we will be loading the data from the path"""
        try:
            self.data = pd.read_csv(self.data_path, low_memory=False)
            print("Data loaded successfully")
        except FileNotFoundError:
            print("Error loading data {self.data_path}")

    def clean_data(self, drop_missing_values=False, replace_inf = True, Fill_value = True):
        """ here we will be cleaning the data"""
        if drop_missing_values:
            self.data.dropna(inplace=True)
        if replace_inf:
            self.data.replace([np.inf, -np.inf], Fill_value, inplace=True)
    def show_features(self):
        """ Display the features of the data"""
        if data is not None:
            print("features in the datasets: ")
            for feature in self.data.columns:
                print(f"-{feature}")
        else: print("No data loaded")

def main():
    """ main function to execute the script """
    data = DataProcessor("balanced_data.csv")
    data.load_data()
    data.show_features()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main()












