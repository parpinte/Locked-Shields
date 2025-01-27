import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class DatasetHandler:
    def __init__(self, file_path=None):
        self.data = None
        self.test_data = None
        self.train_data = None
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        """
        Load the dataset from a file (CSV or other supported formats).

        Parameters:
        file_path (str): Path to the dataset file.
        """
        self.data = pd.read_csv(file_path)
        print("Data loaded successfully. Shape:", self.data.shape)

    def split_data(self, label_column, test_size=0.2, random_state=42):
        """
        Split the dataset into train and test sets while respecting the label distribution.

        Parameters:
        label_column (str): Name of the column containing the labels.
        test_size (float): Proportion of the dataset to include in the test split (default 0.2).
        random_state (int): Random state for reproducibility (default 42).
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Use load_data() to load a dataset first.")

        X = self.data.drop(columns=[label_column])
        y = self.data[label_column]

        # Perform stratified split
        self.train_data, self.test_data = train_test_split(
            self.data, test_size=test_size, stratify=y, random_state=random_state
        )
        print("Data split successfully. Test data shape:", self.test_data.shape)

    def store_test_data(self, output_path):
        """
        Store the test dataset to a file.

        Parameters:
        output_path (str): Path to save the test dataset.
        """
        if self.test_data is None:
            raise ValueError("Test data is not generated. Use split_data() to create the test set first.")

        self.test_data.to_csv(output_path, index=False)
        print("Test data stored successfully at:", output_path)

    def store_train_data(self, data, output_path):
        """
        Store the train dataset to a file.

        Parameters:
        output_path (str): Path to save the train dataset.
        """
        if self.train_data is None:
            raise ValueError("Train data is not generated. Use split_data() to create the train set first.")

        data.to_csv(output_path, index=False)
        print("Train data stored successfully at:", output_path)

    def remove_duplicates_from_train(self):
        """
        Remove duplicates from the training dataset.

        Returns:
        pd.DataFrame: The updated training dataset with duplicates removed.
        """
        if self.train_data is None:
            raise ValueError("Train data is not generated. Use split_data() to create the train set first.")

        before_count = len(self.train_data)
        self.train_data = self.train_data.drop_duplicates()
        after_count = len(self.train_data)
        print(f"Duplicates removed. Rows before: {before_count}, Rows after: {after_count}")
        return self.train_data

    def apply_smote(self, label_column, random_state=42):
        if self.train_data is None:
            raise ValueError("Train data is not generated. Use split_data() to create the train set first.")

        X = self.train_data.drop(columns=[label_column])
        y = self.train_data[label_column]

        unique_classes = y.nunique()
        if unique_classes <= 1:
            raise ValueError(
                f"Cannot apply SMOTE. The target column '{label_column}' must have at least 2 unique classes. Got {unique_classes}."
            )

        print("Checking for missing and invalid values...")
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Missing values detected! Dropping...")
            self.train_data = self.train_data.dropna()
            X = self.train_data.drop(columns=[label_column])
            y = self.train_data[label_column]

        # Handle infinite values
        is_finite = X.replace([float('inf'), float('-inf')], float('nan')).notnull().all(axis=1)
        if not is_finite.all():
            print("Infinite values detected! Dropping relevant rows...")
            self.train_data = self.train_data[is_finite]
            X = self.train_data.drop(columns=[label_column])
            y = self.train_data[label_column]

        # Encode non-numeric columns
        non_numeric_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(non_numeric_columns) > 0:
            print(f"Encoding non-numeric columns: {list(non_numeric_columns)}")
            for col in non_numeric_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        # Scale data using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Combine resampled features and labels
        self.train_data = pd.concat(
            [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[label_column])], axis=1
        )

        print("SMOTE applied successfully. Class distribution after SMOTE:")
        print(self.train_data[label_column].value_counts())
        return self.train_data

    def apply_undersampling(self, label_column, random_state=42):
        """
        Apply undersampling to the training dataset to handle class imbalance.

        Parameters:
        label_column (str): Name of the column containing the labels.
        random_state (int): Random state for reproducibility (default 42).

        Returns:
        pd.DataFrame: The updated training dataset with balanced classes.
        """
        if self.train_data is None:
            raise ValueError("Train data is not generated. Use split_data() to create the train set first.")

        X = self.train_data.drop(columns=[label_column])
        y = self.train_data[label_column]

        # Apply undersampling
        undersampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)

        # Combine resampled features and labels back into a DataFrame
        self.train_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[label_column])], axis=1)

        print("Undersampling applied successfully. Class distribution after undersampling:")
        print(self.train_data[label_column].value_counts())
        return self.train_data

if __name__ == "__main__":
    handler  = DatasetHandler(file_path="../data/ls23pr_v1.csv")
    handler.load_data("../data/balanced_data.csv")
    columns_to_drop = ['Flow ID', 'SrcIP', 'DstIP', 'External_src', 'External_dst', 'Conn_state', 'Segment_src',
                       'Segment_dst', 'Expoid_src', 'Expoid_dst', 'mTimestampStart', 'mTimestampLast','Flow ID','SrcPort']

    handler.data.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
    print(handler.data.shape)
    handler.split_data("Label")
    handler.store_test_data("../data/BigTestSet.csv")
    handler.store_train_data(handler.train_data,    "../data/BigTrainSet.csv")
    # lets start with undersampling
    datas_undersampled = handler.apply_undersampling("Label")
    handler.store_train_data(datas_undersampled, "../data/BigTrainSet_undersampled.csv")
    # let's now apply smote
    smote_data = handler.apply_smote("Label")
    handler.store_train_data(smote_data, "../data/BigTrainSet_smote.csv")









