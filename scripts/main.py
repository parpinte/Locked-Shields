
class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None


    # function to loas the data for that you need to have pandas imported as pd
    def load_data(self):
        """ here we will be loading the data from the path"""
        try:
            self.data = pd.read_csv(self.data_path, low_memory=False)
            print("Data loaded successfully")
        except FileNotFoundError:
            print("Error loading data {self.data_path}")

    def clean_data(self, drop_missing_values=True, replace_inf = True, Fill_value = False):
        """ here we will be cleaning the data"""
        if drop_missing_values:
            self.data.dropna(inplace=True)
        if replace_inf:
            self.data.replace([np.inf, -np.inf], Fill_value, inplace=True)

    def show_features(self):
        """ Display the features of the data"""
        if self.data is not None:
            print("features in the datasets: ")
            for feature in self.data.columns:
                print(f"-{feature}")
        else: print("No data loaded")

    def drop_features(self, columns_to_drop):
        """ Drop specific columns from the data
        args : colums_to_drop (list) : list of columns names to be dropped
        use the function show_features in order to checks the different features presents in the dataset
        """
        if self.data is not None:
            self.data.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
            print("Features dropped successfully")
        else: print("No data loaded. please load the data first")

    def split_data(self, target_column):
        """
        Splits the data into features (X) and target (y).
        Args: target_column (str): The name of the column to be used as the target variable.
        Returns: tuple: X (features), y (target)
        """
        if self.data is not None:
            if target_column in self.data.columns:
                self.X = self.data.drop(columns=[target_column])
                self.y = self.data[target_column]
                print(f"Data split into X (features) and y (target) using '{target_column}' as target.")
            else:
                print(f"Error: Column '{target_column}' not found in the dataset.")

        else:
            print("No data loaded. Please load the data first.")


    def detect_categorical(self, handle_nan="unknown"):
        """
        Detects categorical features in the dataset and identifies NaN values.
        Handles NaN values in the categorical features based on the chosen method.

        Args:
            handle_nan (str): How to handle NaN values in categorical features.
                              Options are "drop", "most_frequent", or "unknown".
        """
        if self.data is not None:
            # Detect categorical features
            categorical_features = self.data.select_dtypes(include=['object', 'category']).columns

            if len(categorical_features) == 0:
                print("No categorical features detected.")
                return

            print("Categorical Features and their NaN Information:")
            for feature in categorical_features:
                total_nan = self.data[feature].isna().sum()
                percentage_nan = (total_nan / len(self.data)) * 100

                print(f"- {feature}:")
                print(f"  NaN Count: {total_nan}")
                print(f"  Percentage of NaNs: {percentage_nan:.2f}%")

                # Handle NaN values based on the chosen option
                if total_nan > 0:  # Only handle if there are NaN values
                    if handle_nan == "drop":
                        self.data.dropna(subset=[feature], inplace=True)
                        print(f"  Action: Dropped rows with NaN in '{feature}'.")

                    elif handle_nan == "most_frequent":
                        most_frequent = self.data[feature].mode()[0]
                        # Assign the result back to the column
                        self.data[feature] = self.data[feature].fillna(most_frequent)
                        print(f"  Action: Replaced NaN with most frequent value '{most_frequent}'.")

                    elif handle_nan == "unknown":
                        # Assign the result back to the column
                        self.data[feature] = self.data[feature].fillna("Unknown")
                        print(f"  Action: Replaced NaN with 'Unknown'.")

                    else:
                        print(f"  Action: Invalid option '{handle_nan}'. No changes made for '{feature}'.")

            print("\nCategorical NaN handling completed.")
        else:
            print("No data loaded. Please load the data first.")

    def detect_numerical_issues(self):
        """
        Detects `NaN` and `inf` values in numerical features.
        Prints only the features with issues, showing their counts and percentages.
        Returns a list of features that have issues.

        Returns:
            list: A list of numerical feature names with issues (NaN or inf values).
        """
        if self.data is not None:
            # Detect numerical features
            numerical_features = self.data.select_dtypes(include=['number']).columns

            if len(numerical_features) == 0:
                print("No numerical features detected.")
                return []

            issues_found = []
            print("Numerical Features with NaN/Inf Issues:")
            for feature in numerical_features:
                # Detect NaN values
                total_nan = self.data[feature].isna().sum()
                percentage_nan = (total_nan / len(self.data)) * 100

                # Detect Inf values
                total_inf = self.data[feature].isin([np.inf, -np.inf]).sum()
                percentage_inf = (total_inf / len(self.data)) * 100

                # Process features with issues
                if total_nan > 0 or total_inf > 0:
                    issues_found.append(feature)
                    print(f"- {feature}:")
                    if total_nan > 0:
                        print(f"  NaN Count: {total_nan}, Percentage: {percentage_nan:.2f}%")
                    if total_inf > 0:
                        print(f"  Inf Count: {total_inf}, Percentage: {percentage_inf:.2f}%")

            if not issues_found:
                print("No NaN or Inf issues found in numerical features.")

            return issues_found
        else:
            print("No data loaded. Please load the data first.")
            return []

    def handle_numerical_issues(self, features, treatment="drop", replacement_value=None):
        """
        Handles numerical features with issues (`NaN` or `inf` values) based on the specified treatment.

        Args:
            features (list): List of numerical feature names to handle.
            treatment (str): The treatment to apply. Options are:
                             - "drop": Drops rows with issues in the specified features.
                             - "mean": Replaces issues with the column mean.
                             - "replace": Replaces issues with the specified `replacement_value`.
            replacement_value (float or int, optional): The value to use when `treatment` is "replace".

        Raises:
            ValueError: If an invalid `treatment` is provided.
        """
        if self.data is None:
            print("No data loaded. Please load the data first.")
            return

        for feature in features:
            if feature not in self.data.columns:
                print(f"Warning: Feature '{feature}' not found in the dataset. Skipping.")
                continue

            # Detect NaN and Inf values
            total_issues = self.data[feature].isna().sum() + self.data[feature].isin([np.inf, -np.inf]).sum()
            if total_issues == 0:
                print(f"No issues found in '{feature}'. Skipping.")
                continue

            print(f"Handling issues in '{feature}':")
            if treatment == "drop":
                # Drop rows with NaN or Inf
                self.data = self.data[~self.data[feature].isin([np.inf, -np.inf])]
                self.data.dropna(subset=[feature], inplace=True)
                print(f"  Action: Dropped rows with issues in '{feature}'.")

            elif treatment == "mean":
                # Replace issues with the column mean
                mean_value = self.data[~self.data[feature].isin([np.inf, -np.inf])][feature].mean()
                self.data[feature] = self.data[feature].replace([np.inf, -np.inf], mean_value)
                self.data[feature] = self.data[feature].fillna(mean_value)  # Corrected here
                print(f"  Action: Replaced issues with mean value '{mean_value:.2f}' in '{feature}'.")

            elif treatment == "replace":
                # Replace issues with a specific value
                if replacement_value is not None:
                    self.data[feature] = self.data[feature].replace([np.inf, -np.inf], replacement_value)
                    self.data[feature] = self.data[feature].fillna(replacement_value)  # Corrected here
                    print(f"  Action: Replaced issues with '{replacement_value}' in '{feature}'.")
                else:
                    print(f"  Error: No replacement value provided for '{feature}'. Skipping.")

            else:
                raise ValueError(f"Invalid treatment option '{treatment}' for '{feature}'.")

        print("\nNumerical issue handling completed.")

    def encode_categorical(self):
        """
        Encodes categorical features in the dataset using label encoding.
        Records the transformation details (original features and their categories)
        for later use in deployment or inverse transformation.
        """
        if self.data is not None:
            # Detect categorical features
            categorical_features = self.data.select_dtypes(include=['object', 'category']).columns

            if len(categorical_features) == 0:
                print("No categorical features detected for encoding.")
                return

            # Initialize or reset the encoding mapping
            self.encoding_mapping = {}

            print("Encoding categorical features using label encoding:")
            for feature in categorical_features:
                # Convert to category type if not already
                self.data[feature] = self.data[feature].astype('category')

                # Store the mapping of categories for the feature
                self.encoding_mapping[feature] = {
                    "categories": list(self.data[feature].cat.categories),
                    "index": self.data.columns.get_loc(feature)
                }

                # Apply label encoding
                self.data[feature] = self.data[feature].cat.codes
                print(f"  Label encoded '{feature}'. Categories: {self.encoding_mapping[feature]['categories']}")

            print("\nCategorical encoding completed. Transformation details recorded.")
        else:
            print("No data loaded. Please load the data first.")


    def scale_features(self, exclude_features=None, scaling_type="normalize"):
        """
        Scales all features in `X` using standardization or normalization, excluding specified features.
        Stores the scaled data in `self.X_scaled` and preserves `self.y`.

        Args:
            exclude_features (list, optional): List of feature names to exclude from scaling.
                                               Default is None (all features are scaled).
            scaling_type (str): The type of scaling to apply. Options are:
                                - "standardize": Scales to have zero mean and unit variance.
                                - "normalize": Scales to have values in the range [0, 1].
        """
        if hasattr(self, "X") and hasattr(self, "y"):
            # Ensure exclude_features is a list
            exclude_features = exclude_features or []

            # Identify features to scale, excluding specified ones
            features_to_scale = [f for f in self.X.columns if f not in exclude_features]

            if len(features_to_scale) == 0:
                print("No features to scale (all excluded).")
                return

            print(f"Scaling the following features using '{scaling_type}': {features_to_scale}")

            # Initialize a copy of X for scaling
            self.X_scaled = self.X.copy()

            if scaling_type == "standardize":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                self.X_scaled[features_to_scale] = scaler.fit_transform(self.X[features_to_scale])
                print("  Standardization completed.")

            elif scaling_type == "normalize":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                self.X_scaled[features_to_scale] = scaler.fit_transform(self.X[features_to_scale])
                print("  Normalization completed.")

            else:
                print(f"  Error: Invalid scaling type '{scaling_type}'. Use 'standardize' or 'normalize'.")
                return

            print("\nScaled data stored in `self.X_scaled`.")
        else:
            print("No feature data (X or y) detected. Please split the data into X and y first.")

    def preprocess(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the data into training, validation, and testing sets.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the training set to use for validation.
            random_state (int): Random state for reproducibility.
        """
        # Use self.data and self.target_column to extract features and target
        X = self.X
        y = self.y

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.1, random_state=42
        )


    def check_data_leakage(self):
        """
        Check for data leakage by comparing the overlap of rows between training, validation, and test sets.
        Raises: ValueError: If data leakage is detected.
        """
        # Combine training and validation sets
        train_val_set = pd.concat([self.X_train, self.X_val], axis=0)

        # Check overlap between training/validation set and test set
        overlap_test = pd.merge(train_val_set, self.X_test, how='inner')

        if not overlap_test.empty:
            print("Data leakage detected! Overlapping rows found between training/validation and test sets.")
        else: print("No data leakage detected.")

    def check_and_remove_duplicates(self, drop_duplicates=True):
        """
        Check for and remove duplicate rows in the dataset.

        Returns:
            int: Number of duplicate rows removed.
        """
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            print(f"Duplicate rows found: {duplicates}")
            if drop_duplicates:
                self.data = self.data.drop_duplicates().reset_index(drop=True)
        else:
            print("No duplicate rows found.")

    def class_dist(self):
        plt.figure(figsize=(6, 4))
        self.y_train.value_counts().plot(kind='bar')
        plt.title("Class Distribution in Training Set")
        plt.show()

    def plot_high_correlation(self, threshold=0.7):
        """
        Plot correlations above a specified threshold, hiding features with low correlations.

        Args:
            threshold (float): Correlation threshold to filter values to display.

        Returns:
            tuple: (matplotlib.figure.Figure, pd.DataFrame)
                   The figure object and the filtered correlation matrix.
        """

        corr_matrix = self.X.corr()

        # Mask the lower triangle for cleaner output
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Filter correlations above the threshold
        filtered_corr = corr_matrix.where((abs(corr_matrix) > threshold) & ~mask)

        # Identify features with at least one high correlation
        features_to_keep = filtered_corr.columns[
            (filtered_corr.abs() > threshold).any(axis=0)
        ]

        # Reduce the matrix to only those features
        reduced_corr = filtered_corr.loc[features_to_keep, features_to_keep]

        # Create the heatmap plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            reduced_corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            cbar=True,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(f"Features with Correlation > {threshold}")

        # Return both the figure and the reduced correlation matrix
        return fig, reduced_corr


    def drop_highly_correlated_features(self, threshold=0.85):
        """
        Drop one feature from each pair of highly correlated features.
        Returns: list: List of features dropped.
        """
        corr_matrix = self.X.corr()
        columns_to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    feature_to_drop = corr_matrix.columns[j]
                    columns_to_drop.add(feature_to_drop)

        self.X.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped features due to high correlation: {columns_to_drop}")
        return list(columns_to_drop)



class MLModel:
    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val):
        """
        Initialize the MLModel class with dataset and target variable.

        Args:
            data (pd.DataFrame): The dataset containing features and target.
            target_column (str): The name of the target column in the dataset.
        """
        self.model = None  # Placeholder for the machine learning model
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test


    def set_model(self, model_class, model_params=None):
        """
        Set the machine learning model to use.

        Args:
            model_class: The class of the machine learning model (e.g., RandomForestClassifier).
            model_params (dict): A dictionary of parameters to pass to the model.
        """
        if model_params is None:
            model_params = {}  # Default to an empty dictionary if no parameters provided
        self.model = model_class(**model_params)


    def train(self):
        """
        Train the machine learning model on the training data.
        """
        if self.model is None:
            raise ValueError("Model not set. Use the 'set_model' method to define a model.")
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Run the 'preprocess' method first.")

        print("Training the model...")
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")

    def validate(self):
        """
        Validate the model on the validation data.
            float: Accuracy score of the model on the validation set.
            ValueError: If the model is not trained or validation data is missing.
        """
        if self.model is None:
            raise ValueError("Model not trained. Use the 'train' method first.")
        if self.X_val is None or self.y_val is None:
            raise ValueError("Validation data not prepared. Run the 'preprocess' method first.")

        print("Validating the model...")
        predictions = self.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, predictions)
        print(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def test(self):
        """
        Test the model on the test data.
        float: Accuracy score of the model on the test set.
        ValueError: If the model is not trained or test data is missing.
        """
        if self.model is None:
            raise ValueError("Model not trained. Use the 'train' method first.")
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not prepared. Run the 'preprocess' method first.")

        print("Testing the model...")
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def plot_confusion_matrix(self, dataset='test'):
        """
        Plot the confusion matrix for the model predictions and return the figure.
        Args:
            dataset (str): Which dataset to evaluate ('test' or 'val').

        Returns:
            matplotlib.figure.Figure: The figure object of the confusion matrix.
        """
        if dataset == 'test':
            X, y = self.X_test, self.y_test
        elif dataset == 'val':
            X, y = self.X_val, self.y_val
        else:
            raise ValueError("Invalid dataset. Use 'test' or 'val'.")

        # Generate predictions
        predictions = self.model.predict(X)

        # Create a figure
        fig, ax = plt.subplots()

        # Plot the confusion matrix
        ConfusionMatrixDisplay.from_predictions(y, predictions, ax=ax)

        # Add a title
        ax.set_title(f"Confusion Matrix ({dataset.capitalize()} Set)")

        # Return the figure
        return fig

    def plot_roc_curve(self, dataset='test'):
        """
        Plot the ROC curve for the model predictions and return the figure
        Returns: matplotlib.figure.Figure: The figure object of the ROC curve.
        """
        if dataset == 'test':
            X, y = self.X_test, self.y_test
        elif dataset == 'val':
            X, y = self.X_val, self.y_val
        else:
            raise ValueError("Invalid dataset. Use 'test' or 'val'.")

        # Check if the model supports predict_proba or decision_function
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)[:, 1]  # Probability of the positive class
        elif hasattr(self.model, "decision_function"):
            y_proba = self.model.decision_function(X)  # Decision scores
        else:
            raise ValueError("The model does not support probability predictions or decision function.")

        # Validate probability predictions
        if not (0 <= y_proba.min() <= 1 and 0 <= y_proba.max() <= 1):
            raise ValueError("Probability predictions are not in the range [0, 1]. Check the model or preprocessing.")

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_proba)

        # Check for trivial ROC curve (perfect separation or single point)
        if len(fpr) == 2 and fpr[0] == 0 and fpr[1] == 1 and tpr[1] == 1:
            print("Warning: ROC curve indicates perfect separation. Check for data leakage or overfitting.")

        # Create a figure
        fig, ax = plt.subplots()

        # Plot ROC curve with label
        display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        display.plot(ax=ax, name=f"ROC Curve ({dataset} set)")

        # Add a title to the plot
        ax.set_title(f"ROC Curve ({dataset.capitalize()} Set)")

        # Return the figure
        return fig

    def plot_precision_recall_curve(self, dataset='test'):
        """
        Plot the precision-recall curve for the model predictions and return the figure.
        Args:
            dataset (str): Which dataset to evaluate ('test' or 'val').

        Returns:
            matplotlib.figure.Figure: The figure object of the precision-recall curve.
        """
        if dataset == 'test':
            X, y = self.X_test, self.y_test
        elif dataset == 'val':
            X, y = self.X_val, self.y_val
        else:
            raise ValueError("Invalid dataset. Use 'test' or 'val'.")

        y_proba = self.model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_proba)

        # Create a figure
        fig, ax = plt.subplots()

        # Plot the precision-recall curve
        display = PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot(ax=ax)

        # Return the figure
        return fig


    def compute_metrics(self, dataset='test'):
        """
        Compute key metrics derived from the confusion matrix:
        Accuracy, Precision, Recall, and F1-Score.
        """
        # Choose the appropriate dataset
        if dataset == 'test':
            X, y = self.X_test, self.y_test
        elif dataset == 'val':
            X, y = self.X_val, self.y_val
        else:
            raise ValueError("Invalid dataset. Use 'test' or 'val'.")

        # Get predictions
        predictions = self.model.predict(X)

        # Compute metrics
        metrics = {
            'Accuracy': accuracy_score(y, predictions),
            'Precision': precision_score(y, predictions, zero_division=0),
            'Recall': recall_score(y, predictions, zero_division=0),
            'F1-Score': f1_score(y, predictions, zero_division=0)
        }

        # Print metrics for quick review
        print(f"Metrics for {dataset} dataset:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        return metrics

    def cross_val_score(self):
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        print("Cross-Validation AUC Scores:", cv_scores)

"""
integrate pylint and pytest gitlab.ci 
"""

""" main code to execute the script"""
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # Import the XGBoost library
    from xgboost import XGBClassifier
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import roc_curve, RocCurveDisplay
    from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import json
    import os
    from sklearn.model_selection import cross_val_score
    import seaborn as sns
    """define the name that will be used in order to save all the related files for your analysis after"""
    ref_name = "DROP_2"
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))


    """ main function to execute the script """

    data = DataProcessor("../data/balanced_data.csv")

    data.load_data()

    columns_to_drop = ['Flow ID', 'SrcIP', 'DstIP', 'External_src', 'External_dst', 'Conn_state', 'Segment_src',
                       'Segment_dst', 'Expoid_src', 'Expoid_dst', 'mTimestampStart', 'mTimestampLast']
    data.drop_features(columns_to_drop=columns_to_drop)
    data.clean_data()
    data.detect_categorical(handle_nan="drop")

    # let's handle numerical datas
    features_issues = data.detect_numerical_issues()
    print("shape of the dataset is : ", data.data.shape)
    data.handle_numerical_issues(features_issues, treatment="drop", replacement_value=None)
    data.check_and_remove_duplicates(drop_duplicates=True)
    # let's encode the categorical datas

    data.encode_categorical()
    # split data into dataset for the training and target
    data.split_data(target_column='Label')
    print(data.data.shape)
    print(data.X.shape)
    print(data.y.shape)
    correlation_threshold = 0.7
    fig_corr, filtered_corr = data.plot_high_correlation(threshold=correlation_threshold)
    dropped_correlated_features = data.drop_highly_correlated_features(threshold=correlation_threshold)
    print(f"dropped features due to high correlation: {dropped_correlated_features}")
    print(data.X.shape)
    print(data.y.shape)
    print()
    print(f"filtered correlated features : {filtered_corr}")
    fig_corr.savefig(f"../plots/{ref_name}_fig_corr.png")


    data.scale_features(exclude_features=None, scaling_type="normalize")

    data.preprocess(test_size=0.2, val_size=0.1, random_state=42)
    data.check_data_leakage()
    """ Testing the MLModel class """
    data.check_and_remove_duplicates(drop_duplicates=True)
    data.check_data_leakage()
    ml = MLModel(data.X_train, data.X_test, data.X_val, data.y_train, data.y_test, data.y_val)

    # Define parameters for the XGBClassifier
    xgb_params = {
        'n_estimators': 20,
        'learning_rate': 0.05,
        'max_depth': 8,
        'random_state': 42,
        # 'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    with open(f"../results/{ref_name}_xgb_params.json", "w") as f:
        json.dump(xgb_params, f, indent=4)

    # Set the model using the dictionary of parameters
    ml.set_model(XGBClassifier, xgb_params)



    ml.train()

    ml.validate()

    ml.test()

    confusion_matrix_plot = ml.plot_confusion_matrix(dataset='test')
    confusion_matrix_plot.savefig(f"../plots/{ref_name}_confusion_matrix.png")


    roc_curve_plot = ml.plot_roc_curve(dataset='test')
    roc_curve_plot.savefig(f"../plots/{ref_name}_roc_curve.png")

    """
    True Negatives (Top-left):
    Label: 0 predicted as 0
    Interpretation: These are the instances where the model correctly predicted class 0.

    False Positives (Top-right):
    Label: 0 predicted as 1
    Interpretation: These are the instances where the model incorrectly predicted class 1 when the true label was 0.

    False Negatives (Bottom-left):
    Label: 1 predicted as 0
    Interpretation: These are the instances where the model incorrectly predicted class 0 when the true label was 1.

    True Positives (Bottom-right):
    Label: 1 predicted as 1
    Interpretation: These are the instances where the model correctly predicted class 1.

    """

    recall_curve_fig = ml.plot_precision_recall_curve(dataset='test')
    recall_curve_fig.savefig(f"../results/{ref_name}_precision_recall_curve.png")
    plt.close(recall_curve_fig)
    metrics = ml.compute_metrics(dataset='test')
    with open(f"../results/{ref_name}_metrics_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    ml.cross_val_score()

    data = data.class_dist()
