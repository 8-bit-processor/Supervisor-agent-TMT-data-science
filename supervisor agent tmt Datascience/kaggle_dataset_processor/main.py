from agents.data_wrangling_agent import DataWranglingAgent
from agents.modeling_agent import ModelingAgent

def process_dataset(df, data_agent, model_agent):
    """
    Handles the data cleaning, exploration, processing, and model training steps.
    """
    if df is None:
        print("Failed to load dataset. Please try again.")
        return

    print("\nDataset loaded successfully!")
    
    # 1. Clean Data
    df = data_agent.clean_data(df)
    
    # 2. Explore Data
    data_agent.explore_dataset(df)
    
    # 3. Get Target and Preprocess for Modeling
    target_column = input("Enter the name of the target column for classification: ").strip()
    
    if target_column in df.columns:
        X, y = data_agent.preprocess_data(df, target_column)

        if X is None:
             print("Data preprocessing failed. Aborting.")
             return

        # 4. Select and Train Model
        print("\nSelect a model to train:")
        print("1. Logistic Regression")
        print("2. Decision Tree Classifier")
        print("3. Random Forest Classifier")
        model_choice = input("Enter your choice (1, 2, or 3): ").strip()

        models = {
            '1': 'logistic_regression',
            '2': 'decision_tree',
            '3': 'random_forest'
        }

        if model_choice in models:
            model_name = models[model_choice]
            model_agent.train_and_evaluate_model(X, y, model_name)
        else:
            print("Invalid model choice.")
    else:
        print(f"Error: Target column '{target_column}' not found in the dataset.")

def main():
    """
    Main function to run the application.
    """
    print("Welcome to the Scikit-learn Model Trainer!")

    # Initialize agents
    data_agent = DataWranglingAgent()
    model_agent = ModelingAgent()

    while True:
        # Main menu
        print("\nWhat would you like to do?")
        print("1. Download a dataset from Kaggle")
        print("2. Load a dataset from a local CSV file")
        print("3. Search for a dataset on Kaggle")
        print("4. Exit")
        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

        df = None
        if choice == '1':
            dataset_name = input("Enter the Kaggle dataset name (e.g., 'user/dataset-name'): ").strip()
            df = data_agent.load_dataset(kaggle_dataset=dataset_name)
            process_dataset(df, data_agent, model_agent)
        elif choice == '2':
            file_path = input("Enter the path to your local CSV file: ").strip()
            df = data_agent.load_dataset(local_path=file_path)
            process_dataset(df, data_agent, model_agent)
        elif choice == '3':
            search_term = input("Enter a search term: ").strip()
            search_results = data_agent.search_datasets(search_term)
            if search_results:
                download_choice = input("\nWould you like to download one of these datasets? (yes/no): ").strip()
                if download_choice.lower() == 'yes':
                    try:
                        num_choice = int(input("Enter the number of the dataset to download: ").strip())
                        if 1 <= num_choice <= len(search_results):
                            dataset_to_download = search_results[num_choice - 1]
                            df = data_agent.load_dataset(kaggle_dataset=dataset_to_download.ref)
                            process_dataset(df, data_agent, model_agent)
                        else:
                            print("Invalid number.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

        another_task = input("\nWould you like to perform another task? (yes/no): ").strip()
        if another_task.lower() != 'yes':
            print("Exiting.")
            break

if __name__ == "__main__":
    main()
