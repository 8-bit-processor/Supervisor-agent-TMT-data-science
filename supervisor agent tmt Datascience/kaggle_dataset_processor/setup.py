
from setuptools import setup, find_packages

setup(
    name='kaggle_dataset_processor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'kaggle',
        'joblib',
        'matplotlib',
        'seaborn',
        'tabulate',
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'kgp=kaggle_dataset_processor.main:main',
        ],
    },
)
