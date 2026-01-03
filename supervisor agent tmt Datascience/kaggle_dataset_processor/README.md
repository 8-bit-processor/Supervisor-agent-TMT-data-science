# Kaggle Dataset Processor and Model Trainer

This project provides a simple framework to download datasets from Kaggle or use local datasets, train popular scikit-learn models, and save the results. It is structured using an agent-based architecture to allow for future expansion into a multi-agent data science platform.

## Features

- **Agent-Based Architecture**:
    - `DataWranglingAgent`: Handles dataset loading and preprocessing.
    - `ModelingAgent`: Handles model training, evaluation, and saving.
- **Dataset Loading**: Load datasets directly from Kaggle or from a local CSV file.
- **Model Training**: Train classification models like Logistic Regression, Decision Tree, and Random Forest.
- **Model Saving**: Save the trained models and their performance reports.
- **Extensible**: Easily extendable to include more models, agents, and preprocessing steps.

## Prerequisites

- Python 3.x
- Kaggle API credentials (`kaggle.json`)

## Setup

1.  **Clone the repository or download the files.**

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Kaggle API credentials:**
    - Go to your Kaggle account page.
    - Click on 'Create New API Token'. This will download a `kaggle.json` file.
    - Place the `kaggle.json` file in the appropriate location. On Windows, this is typically `C:\Users\<Your-Username>\.kaggle\kaggle.json`. On Linux/macOS, it's `~/.kaggle/kaggle.json`.

## How to Use

1.  **Run the main application:**
    ```bash
    python main.py
    ```

2.  **Follow the on-screen prompts:**
    - Choose to download a dataset from Kaggle or use a local CSV file.
        - If using Kaggle, provide the dataset name (e.g., `uciml/iris`).
        - If using a local file, provide the file path.
    - Select a model to train from the list.
    - Enter the name of the target column for classification.

3.  **View the results:**
    - The classification report and confusion matrix will be printed to the console.
    - The trained model will be saved in the `models/` directory.
    - The classification report will be saved as a text file in the `models/` directory.

## Project Structure

```
kaggle_dataset_processor/
├── agents/                # Directory for agent classes
│   ├── data_wrangling_agent.py
│   └── modeling_agent.py
├── data/                  # Directory for storing datasets
├── models/                # Directory for storing trained models and reports
├── main.py                # Main application script (Orchestrator)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```