
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

class ReportingAgent:
    def __init__(self):
        self.reports_dir = 'models' # Save reports in the same directory as models
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)

    def generate_report(self, model, X, y_test, y_pred, model_name):
        """
        Generates and saves a comprehensive report of the model's performance.
        """
        print("\n--- Generating Report ---")
        
        # 1. Display and save classification report
        report_str = self._get_classification_report_table(y_test, y_pred)
        print("\nClassification Report:")
        print(report_str)
        report_path = os.path.join(self.reports_dir, f'{model_name}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_str)
        print(f"Classification report saved to '{report_path}'")

        # 2. Plot and save confusion matrix
        cm_path = os.path.join(self.reports_dir, f'{model_name}_confusion_matrix.png')
        self._plot_confusion_matrix(y_test, y_pred, cm_path)
        print(f"Confusion matrix plot saved to '{cm_path}'")

        # 3. Plot and save feature importance (if applicable)
        if hasattr(model, 'feature_importances_'):
            fi_path = os.path.join(self.reports_dir, f'{model_name}_feature_importance.png')
            self._plot_feature_importance(model, X.columns, fi_path)
            print(f"Feature importance plot saved to '{fi_path}'")
        
        print("--- Report Generation Complete ---")

    def _get_classification_report_table(self, y_test, y_pred):
        """
        Formats the classification report into a nice table.
        """
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        return tabulate(report_df, headers='keys', tablefmt='grid')

    def _plot_confusion_matrix(self, y_test, y_pred, save_path):
        """
        Plots a confusion matrix heatmap and saves it.
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(save_path)
        plt.close() # Close the plot to free up memory

    def _plot_feature_importance(self, model, feature_names, save_path):
        """
        Plots feature importances for tree-based models.
        """
        importances = model.feature_importances_
        indices = pd.Series(importances, index=feature_names).nlargest(15).index
        top_importances = pd.Series(importances, index=feature_names).nlargest(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_importances, y=top_importances.index)
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
