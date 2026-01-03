import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from .reporting_agent import ReportingAgent # Import the new ReportingAgent

class ModelingAgent:
    def __init__(self):
        self.models_dir = 'models'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.reporting_agent = ReportingAgent() # Instantiate the ReportingAgent

    def train_and_evaluate_model(self, X, y, model_name):
        """
        Trains a classification model, evaluates it, and saves it.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        if model_name == 'logistic_regression':
            model = LogisticRegression(max_iter=1000)
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        else:
            print("Invalid model name.")
            return

        # Train model
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print("Training complete.")

        # Evaluate model
        y_pred = model.predict(X_test)

        # --- DELEGATE TO REPORTING AGENT ---
        # Instead of printing and saving here, we call the reporting agent
        self.reporting_agent.generate_report(model, X, y_test, y_pred, model_name)
        
        # --- SAVE THE MODEL ---
        # This part remains here as it's part of the modeling agent's core responsibility
        model_path = os.path.join(self.models_dir, f'{model_name}_model.joblib')
        joblib.dump(model, model_path)
        print(f"\nModel saved to '{model_path}'")

    # The _save_results method is no longer needed here as its logic is now in the ReportingAgent
    # def _save_results(self, model, model_name, report): ...