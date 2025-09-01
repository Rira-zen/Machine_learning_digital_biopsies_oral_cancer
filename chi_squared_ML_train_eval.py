import pandas as pd
from tkinter import Tk, filedialog, simpledialog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import chi2_contingency
import joblib  # For saving trained models


def get_user_inputs():
    """
    Use Tkinter dialogs to prompt the user for inputs.
    """
    root = Tk()
    root.withdraw()  # Hide the root window

    # File selection dialog
    file_path = filedialog.askopenfilename(
        title="Select the Excel File", filetypes=[("Excel Files", "*.xlsx *.xls")]
    )
    if not file_path:
        raise ValueError("No file selected.")

    # Sheet name dialog
    sheet_name = simpledialog.askstring("Input", "Enter the sheet name in the Excel file:")

    # Target class column dialog
    class_column = simpledialog.askstring("Input", "Enter the name of the target/class column:")

    # Number of feature sets dialog
    num_feature_sets = simpledialog.askinteger("Input", "How many feature sets are present? (e.g., 2):")
    feature_sets = {}

    for i in range(1, num_feature_sets + 1):
        feature_set_name = f"Feature Set {i}"
        features = simpledialog.askstring("Input", f"Enter the feature names for {feature_set_name}, separated by commas:")
        feature_sets[feature_set_name] = [feature.strip() for feature in features.split(",")]

    return file_path, sheet_name, class_column, feature_sets


def load_and_preprocess_data(filepath, sheet_name, class_column, feature_sets):
    """
    Load the data from the specified Excel sheet and preprocess it.
    """
    # Load the data
    data = pd.read_excel(filepath, sheet_name=sheet_name)

    # Encode the target class column
    encoder = LabelEncoder()
    data['target'] = encoder.fit_transform(data[class_column])

    # Create a mapping of numeric labels to original class names
    class_mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))  # e.g., {0: "Class 1", 1: "Class 2"}

    # Combine all specified features into a single list
    selected_features = [feature for features in feature_sets.values() for feature in features]

    # Check that all specified features exist in the dataset
    missing_features = [feature for feature in selected_features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following features are missing from the dataset: {missing_features}")

    # Extract only the selected features and the encoded target column
    features = data[selected_features]

    return features, data['target'], class_mapping, data


def chi_square_test(data, target_column, feature_columns):
    """
    Perform chi-square tests for each feature against the target column.
    """
    results = []
    for feature in feature_columns:
        try:
            # Create contingency table and compute chi-squared test
            contingency_table = pd.crosstab(data[target_column], data[feature])
            chi2, p, dof, _ = chi2_contingency(contingency_table)
            results.append({
                "Feature": feature,
                "Chi2": chi2,
                "p-value": p,
                "Degrees of Freedom": dof
            })
        except Exception as e:
            print(f"Error processing feature {feature}: {e}")
    return pd.DataFrame(results)


def train_with_grid_search(X, y, class_mapping, output_dir):
    """
    Train models using grid search for hyperparameter optimization.
    Return best models and their predictions.
    """
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        }),
        "SVM": (SVC(probability=True), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }),
        "RandomForest": (RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None]
        }),
        "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5, 10]
        })
    }

    results = {}
    grid_search_results = {}

    for model_name, (model, param_grid) in models.items():
        print(f"Training {model_name} with grid search...")
        # Define the F1 scorer
        f1_scorer = make_scorer(f1_score, average='weighted')

        # Grid search with cross-validation
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=f1_scorer, cv=5)
        grid.fit(X, y)

        # Best model and hyperparameters
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        print(f"Best parameters for {model_name}: {best_params}")

        # Save grid search results
        grid_search_results[model_name] = pd.DataFrame(grid.cv_results_)

        # Save the trained model
        model_path = f"{output_dir}/{model_name}_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"Saved {model_name} model to {model_path}")

        # Make predictions using the best model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Map numeric labels back to original class names
        y_test_labels = [class_mapping[val] for val in y_test]
        y_pred_labels = [class_mapping[val] for val in y_pred]

        # Save predictions
        results[model_name] = pd.DataFrame({
            "Ground Truth": y_test_labels,
            "Predicted": y_pred_labels
        })

    return results, grid_search_results


def save_predictions_to_excel(results, output_path):
    """
    Save the ground truth and predicted values for each model to an Excel workbook.
    """
    with pd.ExcelWriter(output_path) as writer:
        for model_name, predictions_df in results.items():
            predictions_df.to_excel(writer, sheet_name=f"{model_name}_Predictions", index=False)


def save_grid_search_results(grid_search_results, output_path):
    """
    Save the grid search results to an Excel workbook.
    """
    with pd.ExcelWriter(output_path) as writer:
        for model_name, grid_results_df in grid_search_results.items():
            grid_results_df.to_excel(writer, sheet_name=f"{model_name}_GridSearch", index=False)


def save_chi_square_results(chi_square_results, output_path):
    """
    Save the chi-square test results to an Excel workbook.
    """
    with pd.ExcelWriter(output_path) as writer:
        for feature_set_name, chi_square_df in chi_square_results.items():
            chi_square_df.to_excel(writer, sheet_name=feature_set_name, index=False)


def get_output_directory():
    """
    Prompt the user to select an output directory for saving results.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        raise ValueError("No directory selected. Please select an output directory.")
    return output_dir


def main():
    # Get user inputs
    filepath, sheet_name, class_column, feature_sets = get_user_inputs()

    # Select output directory
    output_dir = get_output_directory()

    # Load and preprocess data
    X, y, class_mapping, original_data = load_and_preprocess_data(filepath, sheet_name, class_column, feature_sets)

    # Perform chi-squared tests on original data for each feature set
    chi_square_results = {}
    for feature_set_name, features in feature_sets.items():
        print(f"Processing chi-squared tests for {feature_set_name}...")
        chi_square_df = chi_square_test(original_data, 'target', features)
        chi_square_results[feature_set_name] = chi_square_df

    # Save chi-square results to an Excel file
    chi_square_output_path = f"{output_dir}/chi_square_results.xlsx"
    save_chi_square_results(chi_square_results, chi_square_output_path)
    print(f"Chi-squared test results saved to {chi_square_output_path}")

    # Train models with grid search and collect predictions
    results, grid_search_results = train_with_grid_search(X, y, class_mapping, output_dir)

    # Save predictions to an Excel file
    predictions_path = f"{output_dir}/model_predictions.xlsx"
    save_predictions_to_excel(results, predictions_path)
    print(f"Predictions have been saved to {predictions_path}")

    # Save grid search results to an Excel file
    grid_search_path = f"{output_dir}/grid_search_results.xlsx"
    save_grid_search_results(grid_search_results, grid_search_path)
    print(f"Grid search results have been saved to {grid_search_path}")


if __name__ == "__main__":
    main()
