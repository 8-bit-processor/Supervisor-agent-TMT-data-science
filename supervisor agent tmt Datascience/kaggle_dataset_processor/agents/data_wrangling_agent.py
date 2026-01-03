import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import LabelEncoder

class DataWranglingAgent:
    def __init__(self):
        self.data_dir = 'data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        try:
            self.api = KaggleApi()
            self.api.authenticate()
        except Exception as e:
            # Handle cases where Kaggle API is not configured, but allow local file operations
            print(f"Kaggle API not configured. Proceeding without Kaggle features. Error: {e}")
            self.api = None

    def search_datasets(self, search_term):
        """
        Searches for datasets on Kaggle and returns the results.
        """
        if not self.api:
            print("Kaggle API is not configured. Cannot perform search.")
            return []
        print(f"\nSearching for datasets matching '{search_term}'...")
        try:
            results = self.api.dataset_list(search=search_term)
            if results:
                print("Found the following datasets (top 10):")
                for i, dataset in enumerate(results[:10]):
                    print(f"  {i+1}. {dataset.ref} (Size: {dataset.total_bytes} bytes, Owner: {dataset.owner_name})")
                return results[:10]
            else:
                print("No datasets found matching your search term.")
                return []
        except Exception as e:
            print(f"An error occurred while searching: {e}")
            return []

    def load_dataset(self, kaggle_dataset=None, local_path=None):
        """
        Loads a dataset from Kaggle or a local path.
        """
        if kaggle_dataset:
            if not self.api:
                print("Kaggle API is not configured. Cannot download dataset.")
                return None
            return self._download_from_kaggle(kaggle_dataset)
        elif local_path:
            return self._load_from_local(local_path)
        else:
            print("No dataset specified.")
            return None

    def _download_from_kaggle(self, kaggle_dataset):
        print(f"Downloading dataset '{kaggle_dataset}' from Kaggle...")
        try:
            self.api.dataset_download_files(kaggle_dataset, path=self.data_dir, unzip=True)
            downloaded_files = os.listdir(self.data_dir)
            csv_files = [f for f in downloaded_files if f.endswith('.csv')]
            if csv_files:
                for f in csv_files[1:]:
                    os.remove(os.path.join(self.data_dir, f))
                dataset_path = os.path.join(self.data_dir, csv_files[0])
                print(f"Dataset downloaded and extracted to '{dataset_path}'")
                return pd.read_csv(dataset_path)
            else:
                print("No CSV file found in the downloaded dataset.")
                return None
        except Exception as e:
            print(f"Error downloading dataset from Kaggle: {e}")
            return None

    def _load_from_local(self, local_path):
        if os.path.exists(local_path) and local_path.endswith('.csv'):
            print(f"Loading dataset from '{local_path}'")
            return pd.read_csv(local_path)
        else:
            print(f"Error: The file '{local_path}' does not exist or is not a CSV file.")
            return None

    def clean_data(self, df):
        """
        Orchestrates the data cleaning process based on user input.
        """
        print("\n--- Data Cleaning ---")
        df = self._remove_duplicates(df)
        df = self._handle_missing_values(df)
        df = self._drop_columns(df)
        print("--- Data Cleaning Complete ---")
        return df

    def _remove_duplicates(self, df):
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"Removed {rows_removed} duplicate rows.")
        else:
            print("No duplicate rows found.")
        return df

    def _handle_missing_values(self, df):
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if not missing_cols.empty:
            print("\nFound missing values in the following columns:")
            print(missing_cols)
            choice = input("How would you like to handle missing values? (drop_rows/impute/skip): ").strip().lower()
            if choice == 'drop_rows':
                df.dropna(inplace=True)
                print("Dropped rows with missing values.")
            elif choice == 'impute':
                for col in missing_cols.index:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        impute_value = df[col].median()
                        df[col].fillna(impute_value, inplace=True)
                        print(f"Imputed missing values in '{col}' with median ({impute_value}).")
                    else:
                        impute_value = df[col].mode()[0]
                        df[col].fillna(impute_value, inplace=True)
                        print(f"Imputed missing values in '{col}' with mode ('{impute_value}').")
            else:
                print("Skipping missing value handling.")
        else:
            print("No missing values found.")
        return df

    def _drop_columns(self, df):
        choice = input("\nWould you like to drop any columns? (yes/no): ").strip().lower()
        if choice == 'yes':
            print("Current columns:")
            for col in df.columns:
                print(f"- {col}")
            cols_to_drop = input("Enter a comma-separated list of columns to drop: ").strip()
            cols_list = [col.strip() for col in cols_to_drop.split(',')]
            existing_cols = [col for col in cols_list if col in df.columns]
            if existing_cols:
                df.drop(columns=existing_cols, inplace=True)
                print(f"Dropped columns: {', '.join(existing_cols)}")
            else:
                print("No valid columns selected to drop.")
        return df

    def explore_dataset(self, df):
        """
        Provides various views of the dataframe to help with understanding.
        """
        if df is None:
            print("No dataset to explore.")
            return
        print("\n--- Dataset Overview ---")
        print("First 5 rows:")
        print(df.head())
        print("\n--- Dataset Info ---")
        df.info()
        print("\n--- Descriptive Statistics ---")
        print(df.describe())
        print("\n--- Column Names ---")
        for col in df.columns:
            print(f"- {col}")
        print("--------------------")

    def preprocess_data(self, df, target_column):
        """
        Prepares the data for modeling (encoding only).
        """
        print("\nPreprocessing data for modeling (encoding)...")
        # Ensure target column exists
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not in DataFrame.")
            return None, None
            
        # Drop rows with missing target values (should be minimal after cleaning)
        df.dropna(subset=[target_column], inplace=True)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # Encode target variable if it's not numeric
        if y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
        
        print("Encoding complete.")
        return X, y
