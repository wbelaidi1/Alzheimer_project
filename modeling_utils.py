import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def build_model_pipeline(preprocessor, model):
    """
    Creates a full pipeline by combining the preprocessor and a model.
    """
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def train_model(pipeline, X_train, y_train):
    """
    Fits the model pipeline to the training data.
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def perform_grid_search(pipeline, param_grid, X_train, y_train, cv=5, scoring='f1'):
    """
    Performs GridSearch to find the best model hyperparameters.
    """
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluates the model on the test set and prints a classification report.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] # For AUC

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Print a summary
    print("--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    return metrics

def save_model(pipeline, filepath):
    """
    Saves the trained model pipeline to a file using joblib.
    """
    joblib.dump(pipeline, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Loads a trained model pipeline from a file using joblib.
    """
    pipeline = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return pipeline
