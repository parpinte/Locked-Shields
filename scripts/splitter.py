import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

class DataSplitter:
    def __init__(self, X=None, y=None):
        """
        Initialize the DataSplitter with the dataset and target variable.
            X (pd.DataFrame, optional): The features dataset.
            y (pd.Series, optional): The target variable.
        """
        self.X = X
        self.y = y
        self.splits = {}  # To store the training, validation, and test sets

    def get_splits(self):
        if not self.splits:
            raise ValueError("No splits available. Please call a split method (e.g., `random_split`) first.")
        return self.splits

    def load_from_csv(self, X_path, y_path):
        """
        Load the feature dataset (X) and target variable (y) from CSV files.
            X_path (str): Path to the CSV file containing the feature dataset.
            y_path (str): Path to the CSV file containing the target variable.
        """
        try:
            self.X = pd.read_csv(X_path)
            self.y = pd.read_csv(y_path).squeeze()  # Ensure y is a Series, not a DataFrame
            print(f"Data loaded successfully:\n- Features: {self.X.shape}\n- Target: {self.y.shape}")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")

    def set_data(self, X, y):
        """
        Set the feature dataset (X) and target variable (y) directly.
            X (pd.DataFrame): Feature dataset.
            y (pd.Series or pd.DataFrame): Target variable.

        """
        # Validate input types
        if not isinstance(X, pd.DataFrame):
            print("Error: X must be a pandas DataFrame.")
            return

        if not isinstance(y, (pd.Series, pd.DataFrame)):
            print("Error: y must be a pandas Series or DataFrame.")
            return

        # Assign values
        self.X = X
        self.y = y.squeeze()  # Convert DataFrame to Series if necessary
        print(f"Data set successfully:\n- Features: {self.X.shape}\n- Target: {self.y.shape}")

    def random_split(self, test_size=0.2, val_size=0.0, random_state=42):
        """
        Perform a random split of the dataset into training, validation, and test sets.
            test_size
            val_size
            random_state (int): Random seed for reproducibility.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data and target are not set. Use `load_from_csv` or `set_data` to initialize them.")

        # Split into train+validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        if val_size > 0:
            # Adjust validation size relative to train+validation
            val_relative_size = val_size / (1 - test_size)

            # Split train+validation into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state
            )

            # Store splits
            self.splits = {
                "train": (X_train, y_train),
                "validation": (X_val, y_val),
                "test": (X_test, y_test),
            }

            print(f"Random split completed:\n- Train: {len(X_train)}\n- Validation: {len(X_val)}\n- Test: {len(X_test)}")
        else:
            # No validation set; store train and test splits only
            self.splits = {
                "train": (X_train_val, y_train_val),
                "test": (X_test, y_test),
            }

            print(f"Random split completed:\n- Train: {len(X_train_val)}\n- Test: {len(X_test)}")

    def chronological_split(self, test_size=0.2, val_size=0.1, use_validation=False):
        """
        Perform a chronological split of the dataset into training, validation, and test sets.
        keep the earliers samples for the train and validation and the later for the test
        please make sure that the data are sorted by date ( timestamp )
            test_size (float)
            val_size (float)
            use_validation (bool)
        """
        if self.X is None or self.y is None:
            raise ValueError("Data and target are not set. Use `load_from_csv` or `set_data` to initialize them.")

        # Total number of samples
        n_samples = len(self.X)

        # Calculate split indices
        test_split_index = int(n_samples * (1 - test_size))
        if use_validation:
            val_split_index = int(test_split_index * (1 - val_size))
        else:
            val_split_index = test_split_index

        # Create splits
        X_train = self.X.iloc[:val_split_index]
        y_train = self.y.iloc[:val_split_index]

        if use_validation:
            X_val = self.X.iloc[val_split_index:test_split_index]
            y_val = self.y.iloc[val_split_index:test_split_index]
        else:
            X_val, y_val = None, None

        X_test = self.X.iloc[test_split_index:]
        y_test = self.y.iloc[test_split_index:]

        # Store splits
        self.splits = {
            "train": (X_train, y_train),
            "validation": (X_val, y_val) if use_validation else None,
            "test": (X_test, y_test),
        }

        print(f"Chronological split completed:")
        print(f"- Train: {len(X_train)}")
        if use_validation:
            print(f"- Validation: {len(X_val)}")
        print(f"- Test: {len(X_test)}")

    def stratified_split(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Perform a stratified split of the dataset into training, validation, and test sets.
            test_size (float): Proportion of the data to include in the test split.
            val_size (float): Proportion of the training data to include in the validation split.
            random_state (int): Random seed for reproducibility.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data and target are not set. Use `load_from_csv` or `set_data` to initialize them.")

        # Split into train+validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        if val_size > 0:
            # Adjust validation size relative to train+validation
            val_relative_size = val_size / (1 - test_size)

            # Split train+validation into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
            )

            # Store splits
            self.splits = {
                "train": (X_train, y_train),
                "validation": (X_val, y_val),
                "test": (X_test, y_test),
            }

            print(f"Stratified split completed:\n- Train: {len(X_train)}\n- Validation: {len(X_val)}\n- Test: {len(X_test)}")
        else:
            # No validation set; store train and test splits only
            self.splits = {
                "train": (X_train_val, y_train_val),
                "test": (X_test, y_test),
            }

            print(f"Stratified split completed:\n- Train: {len(X_train_val)}\n- Test: {len(X_test)}")


    def kfold_split(self, n_splits=5, shuffle=False, random_state=None):
        """
        Perform K-Fold Cross-Validation split on the dataset.
            n_splits (int): Number of folds (k).
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (int): Random seed for reproducibility (used when shuffle=True).

        """
        if self.X is None or self.y is None:
            raise ValueError("Data and target are not set. Use `load_from_csv` or `set_data` to initialize them.")

        # Initialize KFold
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        # Store splits
        self.splits = []
        fold_idx = 1

        for train_index, val_index in kf.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            self.splits.append({
                "train": (X_train, y_train),
                "validation": (X_val, y_val),
            })

            print(f"Fold {fold_idx} created:")
            print(f"- Train: {len(train_index)} samples")
            print(f"- Validation: {len(val_index)} samples")
            fold_idx += 1


    def leave_one_out_split(self):
        """
        Perform Leave-One-Out Cross-Validation (LOOCV) on the dataset.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data and target are not set. Use `load_from_csv` or `set_data` to initialize them.")
        # Initialize LeaveOneOut
        loo = LeaveOneOut()

        # Store splits
        self.splits = []
        fold_idx = 1

        for train_index, val_index in loo.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            self.splits.append({
                "train": (X_train, y_train),
                "validation": (X_val, y_val),
            })

            print(f"Fold {fold_idx} created:")
            print(f"- Train: {len(train_index)} samples")
            print(f"- Validation: 1 sample")
            fold_idx += 1


    def save_splits(self, output_dir, file_format="csv"):
        """
        Save the splits to files in a specified folder.
            output_dir (str): Path to the folder where the splits will be saved.
            file_format (str): File format for saving ('csv' or 'excel').
        """
        if not self.splits:
            raise ValueError("No splits available. Please perform a split method first.")

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save each split
        for split_name, split_data in self.splits.items():
            if split_data is not None:
                X, y = split_data
                if file_format == "csv":
                    X.to_csv(os.path.join(output_dir, f"{split_name}_X.csv"), index=False)
                    y.to_csv(os.path.join(output_dir, f"{split_name}_y.csv"), index=False)
                elif file_format == "excel":
                    X.to_excel(os.path.join(output_dir, f"{split_name}_X.xlsx"), index=False)
                    y.to_excel(os.path.join(output_dir, f"{split_name}_y.xlsx"), index=False)
                else:
                    raise ValueError("Invalid file format. Use 'csv' or 'excel'.")

        print(f"Splits saved successfully in '{output_dir}'.")

