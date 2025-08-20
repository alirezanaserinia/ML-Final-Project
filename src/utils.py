def load_data(file_path):
    # Function to load data from a given file path
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Function to preprocess the data
    # Example preprocessing steps can be added here
    return data.dropna()

def evaluate_model(model, X_test, y_test):
    # Function to evaluate the model's performance
    from sklearn.metrics import accuracy_score
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def save_model(model, file_path):
    # Function to save the trained model to a file
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    # Function to load a trained model from a file
    import joblib
    return joblib.load(file_path)