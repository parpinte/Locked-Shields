



class Processor:
    def __init__(self, filepath= None):
        self.filepath = filepath
        self.y = None
        self.log = None
        self.data = None

    def load_data(self):
        """
        Load dataset from a CSV file.
        """
        try:
            self.data = pd.read_csv(self.filepath, low_memory=False)
            print("Dataset loaded successfully.")
        except FileNotFoundError:
            print("Error: File not found. Please check the file path.")


    def show_features(self):
        """ Display the features of the data. """
        if self.data is not None:
            print("Features in the dataset:")
            feature_list = self.data.columns.tolist()
            for i, feature in enumerate(feature_list, 1):
                print(f"{i}. {feature}")
            print(f"\nTotal features: {len(feature_list)}")
        else:
            print("No data loaded. Please load a dataset first.")

    def drop_features(self, columns_to_drop):
        """
        Drop specific columns from the data.
        columns_to_drop (list): List of column names to be dropped
        Tip: Use the `show_features` function to check the available features in the dataset.
        """
        if self.data is not None:
            existing_columns = set(self.data.columns)
            columns_to_drop = set(columns_to_drop)
            valid_columns = columns_to_drop.intersection(existing_columns)
            missing_columns = columns_to_drop - valid_columns

            if valid_columns:
                self.data.drop(valid_columns, axis=1, inplace=True)
                print(f"Successfully dropped the following columns: {', '.join(valid_columns)}")
            if missing_columns:
                print(f"Warning: The following columns were not found in the dataset and were ignored: {', '.join(missing_columns)}")
        else:
            print("No data loaded. Please load the data first.")

    def extract_target(self, target_column):
        """
        Extract the target column, store it in self.y, and remove it from the dataset.
        Args: target_column (str): Name of the target column to extract.
        """
        if self.data is not None:
            if target_column in self.data.columns:
                self.y = self.data[target_column]
                self.data.drop(target_column, axis=1, inplace=True)
                print(f"Target column '{target_column}' extracted successfully and stored in self.y.")
            else:
                print(f"Error: Column '{target_column}' not found in the dataset.")
        else:
            print("No data loaded. Please load the data first.")

    def detect_categorical(self, handle_nan="unknown"):
        """
        Detects categorical features in the dataset and identifies NaN values.
        Handles NaN values in the categorical features based on the chosen method.
        Ensures target variable is updated if rows are dropped.

        Args:
            handle_nan (str): How to handle NaN values in categorical features.
                              Options are "drop", "most_frequent", or "unknown".
        """
        if self.data is not None and self.y is not None:
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
                        # Identify rows to keep
                        rows_to_keep = ~self.data[feature].isna()
                        self.data = self.data[rows_to_keep]
                        self.y = self.y[rows_to_keep]
                        print(f"  Action: Dropped rows with NaN in '{feature}'.")

                    elif handle_nan == "most_frequent":
                        most_frequent = self.data[feature].mode()[0]
                        self.data[feature] = self.data[feature].fillna(most_frequent)
                        print(f"  Action: Replaced NaN with most frequent value '{most_frequent}'.")

                    elif handle_nan == "unknown":
                        self.data[feature] = self.data[feature].fillna("Unknown")
                        print(f"  Action: Replaced NaN with 'Unknown'.")

                    else:
                        print(f"  Action: Invalid option '{handle_nan}'. No changes made for '{feature}'.")

            print("\nCategorical NaN handling completed.")
        else:
            print("No data loaded or target variable not set. Please load the data and ensure target is separated.")


    def handle_duplicates(self):
        """
        Check for and remove duplicate rows in the dataset.
        Ensures the corresponding values in the target variable are also removed.
        """
        if self.data is not None and self.y is not None:
            # Find duplicate rows
            duplicates = self.data.duplicated()
            num_duplicates = duplicates.sum()

            if num_duplicates > 0:
                print(f"Found {num_duplicates} duplicate rows in the dataset.")

                # Remove duplicate rows and corresponding target values
                non_duplicates = ~duplicates
                self.data = self.data[non_duplicates]
                self.y = self.y[non_duplicates]

                print("Duplicate rows have been removed.")
            else:
                print("No duplicate rows found in the dataset.")
        else:
            print("No data loaded or target variable not set. Please load the data and ensure the target is separated.")


    def encode_categorical(self, method="label"):
        """
        Encode categorical features in the dataset using the specified method.

        Args:
            method (str): Encoding method to use. Options are "label" (Label Encoding) or "onehot" (One-Hot Encoding).
        """
        if self.data is not None:
            # Detect categorical features
            categorical_features = self.data.select_dtypes(include=['object', 'category']).columns

            if len(categorical_features) == 0:
                print("No categorical features to encode.")
                return

            print(f"Encoding categorical features using {method} encoding.")

            if method == "label":
                for feature in categorical_features:
                    le = LabelEncoder()
                    self.data[feature] = le.fit_transform(self.data[feature])
                    print(f"Feature '{feature}' encoded using Label Encoding.")

            elif method == "onehot":
                self.data = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)
                print(f"Features encoded using One-Hot Encoding.")

            else:
                print(f"Error: Invalid encoding method '{method}'. Choose 'label' or 'onehot'.")

            print("Categorical encoding completed.")
        else:
            print("No data loaded. Please load the data first.")

    def study_correlation(self, threshold=0.8):
        """
        Compute the correlation matrix, identify highly correlated features, and return a figure with only those features.

        Args:
            threshold (float): The correlation coefficient threshold for identifying highly correlated features.

        Returns:
            list: A list of tuples containing pairs of features with correlation above the threshold.
            plt.Figure: The filtered correlation matrix heatmap figure.
        """
        if self.data is not None:
            # Compute the correlation matrix
            correlation_matrix = self.data.corr()

            # Identify highly correlated features
            correlated_features = []
            features_to_include = set()
            for i in range(correlation_matrix.shape[0]):
                for j in range(i + 1, correlation_matrix.shape[1]):
                    if abs(correlation_matrix.iloc[i, j]) > threshold:
                        correlated_features.append((correlation_matrix.index[i], correlation_matrix.columns[j]))
                        features_to_include.update([correlation_matrix.index[i], correlation_matrix.columns[j]])

            # Filter the correlation matrix to only include highly correlated features
            filtered_features = list(features_to_include)
            filtered_correlation_matrix = correlation_matrix.loc[filtered_features, filtered_features]

            # Plot the filtered heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(filtered_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title(f"Filtered Correlation Matrix (Threshold > {threshold})")
            plt.tight_layout()
            fig = plt.gcf()

            # Print highly correlated features
            if correlated_features:
                print("Highly correlated features (absolute correlation > threshold):")
                for pair in correlated_features:
                    print(f"{pair[0]} ↔ {pair[1]}")
            else:
                print("No features are highly correlated based on the threshold.")

            return correlated_features, fig
        else:
            print("No data loaded. Please load the data first.")
            return None, None

    def drop_highly_correlated(self, correlated_features):
        """
        Drop one feature from each pair of highly correlated features, excluding self-correlations.
        Args: correlated_features (list): List of tuples containing pairs of highly correlated features
        """
        if self.data is not None:
            if not correlated_features:
                print("No highly correlated features to drop.")
                return

            # Keep track of dropped features to avoid redundancy
            dropped_features = set()

            for feature1, feature2 in correlated_features:
                # Avoid self-correlations
                if feature1 != feature2:
                    # Drop the second feature in the pair if not already dropped
                    if feature2 not in dropped_features:
                        self.data.drop(columns=[feature2], inplace=True)
                        dropped_features.add(feature2)
                        print(f"Dropped feature: {feature2} (correlated with {feature1})")

            print("Highly correlated features have been addressed.")
        else:
            print("No data loaded. Please load the data first.")

    def apply_pca(self, n_components=None, plot_variance=False):
        """
        Apply PCA for dimensionality reduction.

        Args:
            n_components (int or float): Number of principal components to keep.
                                         If float (0 < n_components <= 1), it represents the variance ratio to preserve.
                                         If None, keep all components.
            plot_variance (bool): If True, plots the explained variance ratio for each component.

        Returns: Transformed dataset with reduced dimensions.
        """
        if self.data is not None:
            # Ensure only numeric data is used for PCA
            numeric_data = self.data.select_dtypes(include=["number"])

            if numeric_data.empty:
                print("No numeric data available for PCA.")
                return None

            # Initialize PCA
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(numeric_data)

            # Create a DataFrame for the reduced data
            reduced_df = pd.DataFrame(
                reduced_data,
                columns=[f"PC{i+1}" for i in range(reduced_data.shape[1])]
            )
            print(f"PCA applied. Reduced dataset shape: {reduced_df.shape}")

            # Optionally plot explained variance ratio
            if plot_variance:
                plt.figure(figsize=(8, 5))
                plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                         pca.explained_variance_ratio_.cumsum(), marker="o")
                plt.title("Cumulative Explained Variance by Principal Components")
                plt.xlabel("Number of Principal Components")
                plt.ylabel("Cumulative Explained Variance")
                plt.grid()
                plt.show()

            return reduced_df
        else:
            print("No data loaded. Please load the data first.")
            return None

    def combine_and_replace_correlated_features(self, correlated_features, method="mean"):
        """
        Combine pairs of correlated features by taking their mean or maximum,
        and replace the original features with the new combined features.

        Args:
            correlated_features (list): List of tuples containing pairs of highly correlated features.
            method (str): Method to combine the features. Options are "mean" or "max".

        Returns:
            None
        """
        if self.data is not None:
            if not correlated_features:
                print("No highly correlated features to combine.")
                return

            # Track processed features to ensure proper replacement
            processed_features = set()

            for feature1, feature2 in correlated_features:
                # Avoid self-correlations and redundant processing
                if feature1 != feature2 and (feature1, feature2) not in processed_features and (feature2, feature1) not in processed_features:
                    # Check if both features exist in the dataset
                    if feature1 in self.data.columns and feature2 in self.data.columns:
                        # Combine the features
                        if method == "mean":
                            self.data[f"{feature1}_{feature2}_combined"] = self.data[[feature1, feature2]].mean(axis=1)
                        elif method == "max":
                            self.data[f"{feature1}_{feature2}_combined"] = self.data[[feature1, feature2]].max(axis=1)
                        else:
                            print(f"Invalid method '{method}'. Use 'mean' or 'max'.")
                            return

                        # Remove the original features
                        self.data.drop(columns=[feature1, feature2], inplace=True)
                        print(f"Replaced '{feature1}' and '{feature2}' with '{feature1}_{feature2}_combined'.")

                        # Mark the pair as processed
                        processed_features.add((feature1, feature2))
                    else:
                        print(f"Skipped combination for '{feature1}' and '{feature2}' as one or both are missing in the dataset.")

            print("Correlated feature replacement completed. Dataset dimensions reduced.")
        else:
            print("No data loaded. Please load the data first.")


    def select_features_by_importance(self, threshold=0.01, model=None):
        """
        Select features based on their importance scores.

        Args:
            threshold (float): Minimum importance score for a feature to be selected.
            model: Pre-trained model with `feature_importances_` attribute.
                   If None, a RandomForestClassifier is used.

        Returns:
            pd.DataFrame: Dataset with selected features only.
        """
        if self.data is not None and self.y is not None:
            # Ensure only numeric data is used for feature selection
            numeric_data = self.data.select_dtypes(include=["number"])

            if numeric_data.empty:
                print("No numeric features available for feature selection.")
                return None

            # Default to a RandomForestClassifier if no model is provided
            if model is None:
                model = RandomForestClassifier(random_state=42)
                model.fit(numeric_data, self.y)

            # Check if the model has the feature_importances_ attribute
            if not hasattr(model, "feature_importances_"):
                print("The provided model does not support feature importance scoring.")
                return None

            # Get feature importances
            feature_importances = model.feature_importances_
            important_features = numeric_data.columns[feature_importances >= threshold]

            print("Selected Features Based on Importance:")
            for feature in important_features:
                print(f"- {feature} (Importance: {feature_importances[numeric_data.columns.get_loc(feature)]:.4f})")

            # Reduce the dataset to only the selected features
            self.data = self.data[important_features]
            print(f"Feature selection completed. Reduced dataset shape: {self.data.shape}")

            return self.data
        else:
            print("No data or target variable loaded. Please load the data and ensure the target is separated.")
            return None

    def detect_missing_numerical(self):
        """
        Detect missing values (NaN and inf) in numerical features of the dataset.

        Prints the number and percentage of missing values for each numerical feature.

        Returns:
            list: A list of feature names with missing values (NaN or inf).
        """
        if self.data is not None:
            # Select numerical features
            numeric_features = self.data.select_dtypes(include=["number"]).columns

            if len(numeric_features) == 0:
                print("No numerical features detected.")
                return []

            missing_features = []

            print("Missing values for numerical features:")
            for feature in numeric_features:
                # Count NaN values
                nan_count = self.data[feature].isna().sum()

                # Count inf values
                inf_count = np.isinf(self.data[feature]).sum()

                if nan_count > 0 or inf_count > 0:
                    # Add feature to the list if it has missing values
                    missing_features.append(feature)

                    # Calculate percentages
                    total_rows = len(self.data)
                    nan_percentage = (nan_count / total_rows) * 100
                    inf_percentage = (inf_count / total_rows) * 100

                    # Print information
                    print(f"- {feature}:")
                    print(f"  NaN Count: {nan_count} ({nan_percentage:.2f}%)")
                    print(f"  Inf Count: {inf_count} ({inf_percentage:.2f}%)")

            if not missing_features:
                print("No missing values detected in numerical features.")

            return missing_features
        else:
            print("No data loaded. Please load the data first.")
            return []


    def handle_missing_values(self, features, method="mean", custom_value=None):
        """
        Handle missing values (NaN and inf) in the specified numerical features.

        Args:
            features (list): List of feature names with missing values to be treated.
            method (str): Method to handle missing values. Options are:
                          - "drop": Drop rows with missing values in the specified features.
                          - "mean": Replace missing values with the mean of the feature.
                          - "custom": Replace missing values with a custom value.
            custom_value (float): The value to replace missing values if method is "custom".

        Returns:
            None
        """
        if self.data is not None and self.y is not None:
            if not features:
                print("No features provided for handling missing values.")
                return

            for feature in features:
                if feature not in self.data.columns:
                    print(f"Feature '{feature}' not found in the dataset. Skipping.")
                    continue

                # Handle based on the specified method
                if method == "drop":
                    rows_to_keep = ~self.data[feature].isna() & ~np.isinf(self.data[feature])
                    print(f"Dropping rows with missing values in '{feature}'.")
                    self.data = self.data[rows_to_keep]
                    self.y = self.y[rows_to_keep]

                elif method == "mean":
                    mean_value = self.data[feature][~self.data[feature].isna() & ~np.isinf(self.data[feature])].mean()
                    self.data[feature] = self.data[feature].replace([np.inf, -np.inf], np.nan).fillna(mean_value)
                    print(f"Replaced missing values in '{feature}' with the mean ({mean_value:.4f}).")

                elif method == "custom":
                    if custom_value is None:
                        print(f"Custom value not provided for '{feature}'. Skipping.")
                        continue
                    self.data[feature] = self.data[feature].replace([np.inf, -np.inf], np.nan).fillna(custom_value)
                    print(f"Replaced missing values in '{feature}' with custom value ({custom_value}).")

                else:
                    print(f"Invalid method '{method}' specified. Use 'drop', 'mean', or 'custom'.")
                    return

            print("Missing value handling completed.")
        else:
            print("No data or target variable loaded. Please load the data and ensure the target is separated.")



    def rescale_data(self, method="standardize"):
        """
        Rescale numerical features in the dataset using standardization or normalization.

        Args:
            method (str): Rescaling method. Options are:
                          - "standardize": Standardize the data (mean=0, std=1).
                          - "normalize": Normalize the data (min=0, max=1).

        Returns:
            None
        """
        if self.data is not None:
            # Ensure only numeric data is rescaled
            numeric_features = self.data.select_dtypes(include=["number"]).columns

            if len(numeric_features) == 0:
                print("No numerical features detected for rescaling.")
                return

            # Select the rescaling method
            if method == "standardize":
                scaler = StandardScaler()
                print("Applying standardization (mean=0, std=1).")
            elif method == "normalize":
                scaler = MinMaxScaler()
                print("Applying normalization (min=0, max=1).")
            else:
                print(f"Invalid method '{method}'. Use 'standardize' or 'normalize'.")
                return

            # Apply the scaler and update the dataset
            self.data[numeric_features] = scaler.fit_transform(self.data[numeric_features])
            print(f"Rescaling completed using {method}.")
        else:
            print("No data loaded. Please load the data first.")


    def summarize_feature_distribution(self, top_categories=5):
        """
        Summarize the distribution of features in the dataset.
        For numerical features, provides summary statistics (mean, std, min, max).
        For categorical features, lists the top categories by count.
        Args:top_categories (int): Number of top categories to display for categorical features.
        Returns: dict: A summary dictionary containing information about numerical and categorical features.
        """
        if self.data is not None:
            summary = {"numerical": {}, "categorical": {}}

            for feature in self.data.columns:
                # Numerical features
                if self.data[feature].dtype in ["int64", "float64"]:
                    stats = self.data[feature].describe()
                    summary["numerical"][feature] = {
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "min": stats["min"],
                        "max": stats["max"],
                    }

                # Categorical features
                elif self.data[feature].dtype == "object" or self.data[feature].dtype.name == "category":
                    value_counts = self.data[feature].value_counts().head(top_categories)
                    summary["categorical"][feature] = value_counts.to_dict()

            # Print the summary in a readable format
            print("Summary of Feature Distribution:")
            print("\nNumerical Features:")
            for feature, stats in summary["numerical"].items():
                print(f"  - {feature}:")
                print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

            print("\nCategorical Features:")
            for feature, categories in summary["categorical"].items():
                print(f"  - {feature}:")
                for category, count in categories.items():
                    print(f"    {category}: {count}")

            return summary
        else:
            print("No data loaded. Please load the data first.")
            return None

    # to test
    def remove_quasi_constant_features(self, threshold=0.99): # function to test
        """
        Detect and remove quasi-constant features from the dataset.

        Args:
            threshold (float): The threshold for detecting quasi-constant features.
                               A feature is considered quasi-constant if the most
                               frequent value appears in more than `threshold` proportion
                               of the rows
        """
        if self.data is not None and self.y is not None:
            # Initialize a list to store quasi-constant features
            quasi_constant_features = []

            # Loop through features to calculate the proportion of the most frequent value
            for feature in self.data.columns:
                # Proportion of the most frequent value
                most_frequent_value_ratio = self.data[feature].value_counts(normalize=True).max()
                if most_frequent_value_ratio >= threshold:
                    quasi_constant_features.append(feature)

            if quasi_constant_features:
                print("Detected quasi-constant features:")
                for feature in quasi_constant_features:
                    print(f"  - {feature} (most frequent value ratio: {most_frequent_value_ratio:.2f})")

                # Drop quasi-constant features
                self.data.drop(columns=quasi_constant_features, inplace=True)
                print(f"Dropped {len(quasi_constant_features)} quasi-constant features.")
            else:
                print("No quasi-constant features detected.")

   #""" these functions are used now to store the different log processes that have been done on the data
   #this helps on knowing on which type of data we are working on"""


    def log_processing_step(self, step_description):
        """
        Log a processing step for tracking data transformations.
        Args: step_description (str): A description of the processing step performed.
        """
        if not hasattr(self, 'log'):
            self.log = []  # Initialize the log if it doesn't exist

        # Append the step description with a timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {step_description}"
        self.log.append(log_entry)
        print(f"Logged step: {log_entry}")

    def show_logs(self):
        """
        Display all logged processing steps.
        """
        if hasattr(self, 'log') and self.log:
            print("\nProcessing Log:")
            for entry in self.log:
                print(entry)
        else:
            print("No processing steps logged yet.")

    def save_logs(self, filepath):
        """
        Save the processing log to a file.

        Args:
            filepath (str): Path to the file where the log will be saved.
        """
        if hasattr(self, 'log') and self.log:
            try:
                with open(filepath, 'w') as log_file:
                    log_file.write("\n".join(self.log))
                print(f"Processing log saved to {filepath}.")
            except Exception as e:
                print(f"Failed to save log: {e}")
        else:
            print("No processing steps logged yet.")

    def study_correlation_with_target(self, target=None, threshold=0.1):
        """
        Analyze the correlation of numerical features with the target variable.

        Args:
            target (pd.Series): The target variable (if separate from the dataset).
                                If None, the class's `self.y` will be used.
            threshold (float): Minimum absolute correlation value to consider a feature relevant.

        Returns:
            pd.DataFrame: A DataFrame with features and their correlation with the target.
        """
        if self.data is not None:
            if target is None:
                if hasattr(self, 'y'):
                    target = self.y
                else:
                    print("Target variable not provided and `self.y` is undefined.")
                    return None

            # Ensure the target is numeric for correlation computation
            if not pd.api.types.is_numeric_dtype(target):
                print("Target variable must be numeric for correlation analysis.")
                return None

            # Compute correlations
            numeric_features = self.data.select_dtypes(include=["number"]).columns
            correlation_results = {}

            for feature in numeric_features:
                corr = self.data[feature].corr(target)
                correlation_results[feature] = corr

            # Convert to DataFrame and filter by threshold
            correlation_df = pd.DataFrame(list(correlation_results.items()), columns=["Feature", "Correlation"])
            correlation_df["Absolute Correlation"] = correlation_df["Correlation"].abs()
            correlation_df = correlation_df.sort_values(by="Absolute Correlation", ascending=False)

            # Filter by threshold
            relevant_features = correlation_df[correlation_df["Absolute Correlation"] >= threshold]

            # Print relevant features
            print(f"Features with absolute correlation >= {threshold}:")
            print(relevant_features)

            return relevant_features
        else:
            print("No data loaded. Please load the data first.")
            return None

    def export_preprocessed_data(self, main_folder="data", subfolder_name=None, file_format="csv"):
        """
        Export the cleaned and transformed dataset, target variable, and logs to a new directory.

        Args:
            main_folder (str): The main directory where subfolders will be created for each export.
            subfolder_name (str): Name of the subfolder for the current export. If None, a timestamp will be used.
            file_format (str): File format for saving the dataset and target. Options: "csv" or "excel".
        """
        # Create the main folder if it doesn't exist
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)

        # Generate subfolder name if not provided
        if subfolder_name is None:
            subfolder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(main_folder, subfolder_name)

        # Create the subfolder
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        try:
            # Save the dataset
            if self.data is not None:
                if file_format == "csv":
                    self.data.to_csv(os.path.join(export_path, "processed_data.csv"), index=False)
                    print(f"Dataset saved in '{export_path}/processed_data.csv'.")
                elif file_format == "excel":
                    self.data.to_excel(os.path.join(export_path, "processed_data.xlsx"), index=False)
                    print(f"Dataset saved in '{export_path}/processed_data.xlsx'.")
                else:
                    print(f"Invalid file format '{file_format}'. Use 'csv' or 'excel'.")
            else:
                print("No dataset available to export.")

            # Save the target variable
            if hasattr(self, 'y') and self.y is not None:
                if file_format == "csv":
                    self.y.to_csv(os.path.join(export_path, "processed_target.csv"), index=False, header=["Target"])
                    print(f"Target variable saved in '{export_path}/processed_target.csv'.")
                elif file_format == "excel":
                    self.y.to_excel(os.path.join(export_path, "processed_target.xlsx"), index=False, header=["Target"])
                    print(f"Target variable saved in '{export_path}/processed_target.xlsx'.")

            # Save the log
            if hasattr(self, 'log') and self.log:
                with open(os.path.join(export_path, "processing_log.txt"), "w") as log_file:
                    log_file.write("\n".join(self.log))
                print(f"Processing log saved in '{export_path}/processing_log.txt'.")
            else:
                print("No log available to export.")

            print(f"Data export completed successfully. Files saved in '{export_path}'.")

        except Exception as e:
            print(f"An error occurred during export: {e}")




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




class MLModel:
    def __init__(self):
        """
        Initialize the MLmodel class."""
        self.model = None
    def set_model(self, model):
        """set a machine learning model"""
        if model is None:
            raise ValueError("The model cannot be None. Please provide a valid machine learning model.")

        self.model = model
        print(f"Model set successfully: {model.__class__.__name__}")

    def train(self, X_train, y_train):
        """
        Train the machine learning model on the provided training data.
            X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
            y_train (pd.Series or np.ndarray): Target vector for training.
        """
        if self.model is None:
            raise ValueError("No model is set. Use `set_model` to specify a model before training.")

        try:
            # Train the model
            self.model.fit(X_train, y_train)
            print("Model trained successfully.")
        except Exception as e:
            print(f"An error occurred during training: {e}")

    def evaluate(self, X_test, y_test, metrics=['accuracy']):
        """
        Evaluate the model's performance on the test dataset.
        Args:
            X_test (pd.DataFrame or np.ndarray): Feature matrix for testing.
            y_test (pd.Series or np.ndarray): Target vector for testing.
            metrics (list): List of metrics to compute. Supported values:
                            'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix'.
        """
        if self.model is None:
            raise ValueError("No model is set. Use `set_model` to specify a model before evaluation.")

        if not hasattr(self.model, 'predict'):
            raise ValueError("The model does not support prediction. Ensure the model is trained and supports the `predict` method.")

        try:
            # Get predictions
            y_pred = self.model.predict(X_test)

            # If the model outputs probabilities, convert them to binary predictions for metrics
            if hasattr(self.model, "predict_proba") and 'roc_auc' in metrics:
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = None

            # Compute metrics
            results = {}
            if 'accuracy' in metrics:
                results['accuracy'] = accuracy_score(y_test, y_pred)
            if 'precision' in metrics:
                results['precision'] = precision_score(y_test, y_pred, average='weighted')
            if 'recall' in metrics:
                results['recall'] = recall_score(y_test, y_pred, average='weighted')
            if 'f1' in metrics:
                results['f1'] = f1_score(y_test, y_pred, average='weighted')
            if 'roc_auc' in metrics and y_pred_proba is not None:
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
            if 'confusion_matrix' in metrics:
                results['confusion_matrix'] = confusion_matrix(y_test, y_pred)

            # Print results
            print("Evaluation Results:")
            for metric, value in results.items():
                print(f"{metric}: {value}")

            return results

        except Exception as e:
            print(f"An error occurred during evaluation: {e}")

    def predict(self, X, proba=False):
        """
        Generate predictions for the given feature matrix.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix for prediction.
            proba (bool): If True, returns probabilities. Otherwise, returns class labels.

        Returns:
            np.ndarray: Predictions (class labels) or probabilities.
        """
        if self.model is None:
            raise ValueError("No model is set. Use `set_model` to specify a model before prediction.")

        if not hasattr(self.model, 'predict'):
            raise ValueError("The model does not support prediction. Ensure the model is trained and supports the `predict` method.")

        if proba:
            # Check if the model supports probability predictions
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X)

                # For binary classification, return the probabilities for the positive class
                if probabilities.shape[1] == 2:
                    return probabilities[:, 1]

                # For multi-class classification, return all probabilities
                return probabilities
            else:
                raise ValueError("The model does not support probability predictions.")
        else:
            # Generate class label predictions
            return self.model.predict(X)




    def get_metrics(self, y_true, y_pred, y_proba=None):
        """
        Compute accuracy, precision, recall, F1 score, and ROC-AUC (if probabilities are provided).

        Args:
            y_true (np.ndarray or pd.Series): Ground truth target values.
            y_pred (np.ndarray or pd.Series): Predicted target values (class labels).
            y_proba (np.ndarray, optional): Predicted probabilities (required for ROC-AUC).

        Returns:
            dict: A dictionary containing accuracy, precision, recall, F1 score, and ROC-AUC (if applicable).
        """
        # Compute core metrics
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }

        # Add ROC-AUC if probabilities are provided and it's binary classification
        if y_proba is not None and len(set(y_true)) == 2:
            results["roc_auc"] = roc_auc_score(y_true, y_proba)

        # Print the metrics
        print("Computed Metrics:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        return results



    def plot_confusion_matrix(self, y_true, y_pred, labels=None, normalize=False, cmap="Blues"):
        """
        Plot a confusion matrix and return the figure.

        Args:
            y_true (np.ndarray or pd.Series): Ground truth target values.
            y_pred (np.ndarray or pd.Series): Predicted target values (class labels).
            labels (list, optional): List of class labels to display on the axes. If None, uses unique values from `y_true`.
            normalize (bool): If True, normalize the confusion matrix by row.
            cmap (str): Colormap for the heatmap.

        Returns:
            matplotlib.figure.Figure: The generated confusion matrix figure.
        """
        # If labels are not provided, use the unique values from y_true
        if labels is None:
            labels = sorted(set(y_true))

        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

        # Return the figure
        return fig


    def plot_roc_auc(self, y_true, y_proba, labels=None, average="macro"):
        """
        Plot the ROC curve and calculate the AUC for binary or multi-class classification.
        Args:
            y_true (np.ndarray or pd.Series): Ground truth target values.
            y_proba (np.ndarray): Predicted probabilities.
            labels (list, optional): List of class labels. If None, inferred from `y_true`.
            average (str): Averaging method for multi-class AUC ('macro', 'weighted').

        """
        # Handle binary classification
        if len(set(y_true)) == 2:
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)

            # Plot ROC curve
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random guess")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right")
            plt.grid()

            return fig

        # Handle multi-class classification
        else:
            if labels is None:
                labels = sorted(set(y_true))

            # Binarize the labels
            y_true_binarized = label_binarize(y_true, classes=labels)

            # Compute ROC curve and AUC for each class
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, label in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
                auc = roc_auc_score(y_true_binarized[:, i], y_proba[:, i])
                ax.plot(fpr, tpr, label=f"Class {label} (AUC = {auc:.2f})")

            # Calculate macro-average AUC
            macro_auc = roc_auc_score(y_true_binarized, y_proba, average=average)
            ax.plot([0, 1], [0, 1], "k--", label=f"Macro-average (AUC = {macro_auc:.2f})")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Multi-Class ROC Curve")
            ax.legend(loc="lower right")
            plt.grid()

            return fig


    def plot_precision_recall_curve(self, y_true, y_proba, positive_label=1):
        """
        Plot the Precision-Recall curve and calculate the Average Precision (AP) score.

        Args:
            y_true (np.ndarray or pd.Series): Ground truth binary target values.
            y_proba (np.ndarray): Predicted probabilities for the positive class.
            positive_label (int): The label of the positive class (default: 1).

        Returns:
            matplotlib.figure.Figure: The generated Precision-Recall curve figure.
        """
        # Compute precision and recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=positive_label)

        # Calculate Average Precision (AP) score
        ap_score = average_precision_score(y_true, y_proba)

        # Plot the Precision-Recall curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f"AP = {ap_score:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="best")
        plt.grid()












if __name__ == "__main__":
    from scipy.signal import correlate
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, label_binarize
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import (
        train_test_split,
        StratifiedShuffleSplit,
        KFold,
        LeaveOneOut,
    )
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
        log_loss,
        matthews_corrcoef,
        balanced_accuracy_score,
        cohen_kappa_score,
        hamming_loss,
        precision_recall_curve,
        average_precision_score,
        roc_curve,
    )
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime
    from xgboost import XGBClassifier




    # Initialize Processor and load data
    Processor = Processor(filepath="../data/BigTrainSet_undersampled.csv")
    Processor.load_data()
    Processor.show_features()

    columns_to_drop = ['Flow ID', 'SrcIP', 'DstIP', 'External_src', 'External_dst', 'Conn_state', 'Segment_src',
                       'Segment_dst', 'Expoid_src', 'Expoid_dst', 'mTimestampStart', 'mTimestampLast']

    Processor.drop_features(columns_to_drop = columns_to_drop)
    Processor.show_features()

    Processor.extract_target(target_column='Label')
    print(Processor.y.value_counts())
    Processor.show_features()

    Processor.detect_categorical(handle_nan="drop")
    Processor.encode_categorical(method="label")

    Processor.handle_duplicates()
    print(Processor.y.value_counts())

    num_issue_features = Processor.detect_missing_numerical()
    Processor.handle_missing_values(num_issue_features, method="drop", custom_value=None)
    print(Processor.y.value_counts())

    Processor.rescale_data(method="standardize")
    Processor.show_logs()
    Processor.summarize_feature_distribution(top_categories=1)

    # Initialize DataSplitter
    splitter = DataSplitter()
    splitter.set_data(Processor.data, Processor.y)
    splitter.random_split(test_size=0.2)

    # test a model

    model = MLModel()
    from sklearn.ensemble import RandomForestClassifier
    model.set_model(RandomForestClassifier())
    train_X, train_y = splitter.splits["train"]
    model.train(train_X, train_y)

    test_X, test_y = splitter.splits["test"]
    results = model.evaluate(test_X, test_y, metrics=["accuracy", "precision", "recall", "f1"])

    y_pred = model.predict(test_X)
    y_proba = model.predict(test_X, proba=True)

    model.plot_roc_auc(test_y, y_proba)














    """
    
    
    
    
    # Initialize DataSplitter
    splitter = DataSplitter()
    splitter.load_from_csv("features.csv", "target.csv")
    splitter.random_split(test_size=0.2)
    
    # Train a model
    model = MLModel()
    from sklearn.ensemble import RandomForestClassifier
    model.set_model(RandomForestClassifier())
    train_X, train_y = splitter.splits["train"]
    model.train(train_X, train_y)
    
    # Evaluate the model
    test_X, test_y = splitter.splits["test"]
    results = model.evaluate(test_X, test_y, metrics=["accuracy", "precision", "recall", "f1"])
    """