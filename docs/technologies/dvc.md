# DVC (Data Version Control)

**DVC** is an open-source version control system for managing data and machine learning models. It integrates with Git and enables versioning of data, models, and code.

DVC is used in our pipeline to manage datasets, store data versions in cloud storage (like AWS S3), and track changes to the datasets that impact the model performance.

### Key Features:
- **Data Versioning**: Allows versioning of large datasets.
- **Data Pipelines**: Enables pipeline management and reproducibility.
- **Storage Integration**: Supports integration with various storage backends like S3, GCP, and Azure.

DVC helps us track data dependencies, ensuring that the model training process can be reproduced at any point in the future.

### Example:
```bash
# Initialize DVC
dvc init

# Add dataset to DVC and push to remote storage
dvc add data/raw/dataset.csv
dvc push
