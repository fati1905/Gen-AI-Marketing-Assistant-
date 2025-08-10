# This file aims to train  a random forest model to classify the output of each client

# IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib


#  Function to split the dataset into X, Y
def split_x_y(df, target):
    """
    This function aims to split a dataset into train and test.

    :param df: The dataframe
    :return:
    X: The features
    Y: The target
    """
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Fit and evaluate the model
def fit_evaluate_model(X_train, X_test, y_train, y_test):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return model:
    """
    # Intialize the model
    model = RandomForestClassifier(
        class_weight="balanced",  # Useful for imbalanced datasets
        n_estimators=100,
        random_state=42
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # For ROC AUC

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    return model


# Save the model
def save_model(model):
    """

    :param model: The model to save
    """
    joblib.dump(model, "/Users/mac-FBENKA22/Desktop/genAI-marketing-assistant/models/random_forest_model.joblib")


def train_model(df, target):
    """
    This function splits the dataset and trains and evaluates a Randomforest model.

    :param df: The dataframe
    :return model: A trained model
    """

    # Split data
    X_train, X_test, y_train, y_test = split_x_y(df, target)

    # Train and evaluate the model
    model = fit_evaluate_model(X_train, X_test, y_train, y_test)

    # Save the model
    save_model(model)

    return model
