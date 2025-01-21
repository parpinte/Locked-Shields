
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns



class MLModel:
    def __init__(self):
        """
        Initialize the MLmodel class."""
        self.model = None

    def set_model(self, model_class, model_params):
        """
        Sets the machine learning model using the model class and parameters.

        Args:
            model_class: The class or function for the ML model you want to use (e.g., XGBClassifier).
            model_params: A dictionary of parameters to initialize the model.
        """
        # Ensure model_class is callable and model_params is a dictionary
        if not callable(model_class):
            raise TypeError("The first argument must be a callable model class or function.")
        if not isinstance(model_params, dict):
            raise TypeError("The second argument must be a dictionary of parameters.")

        # Instantiate the model with the provided parameters
        self.model = model_class(**model_params)

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

