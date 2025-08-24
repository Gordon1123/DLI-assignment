## Setup and Load

    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Import libraries
    import pandas as pd
    
    # Load dataset (adjust path if needed)
    file_path = '/content/drive/My Drive/Colab Notebooks/dataset_full.csv'
    df = pd.read_csv(file_path)
This mounts your Drive so Colab can access files, loads the pandas library, and reads dataset_full.csv into a DataFrame df. After this, your data lives in memory and is ready for cleaning and analysis.

## Core Cleaning
    # Remove duplicates
    print("Duplicate rows before cleaning:", df.duplicated().sum())
    df_clean = df.drop_duplicates()
    
    # Handle missing values
    print("Missing values before cleaning:", df_clean.isnull().sum().sum())
    df_clean = df_clean.fillna(0)   # quick default; swap for median/mode if preferred
This block first reports how many fully duplicated rows exist and removes them, producing df_clean. Then it counts total missing cells and replaces all NaN values with 0 to ensure a complete, model-friendly table.

## Validate and Save

    # Confirm shape change
    print("Shape before:", df.shape)
    print("Shape after :", df_clean.shape)
    
    # Save cleaned dataset back to Drive
    output_path = '/content/drive/My Drive/Colab Notebooks/dataset_full_clean.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")
Here you verify the impact of cleaning by comparing the (rows, columns) before vs. after (rows often shrink due to deduplication). Finally, you write the cleaned DataFrame back to Drive as dataset_full_clean.csv and print the exact save path for quick access.

# Data Preprocessing

## Setup and Load

    # ==========================
    # Data Preparation
    # ==========================
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load the cleaned dataset
    file_path = '/content/drive/My Drive/Colab Notebooks/dataset_full_clean.csv'
    df = pd.read_csv(file_path)
This block imports pandas plus the scikit-learn utilities you’ll use, then loads the already-cleaned CSV into df. From here on, all operations happen in memory. Using the cleaned file ensures duplicates/NaNs are already handled before modeling.

## Features and Split

    # Separate features (X) and label (y)
    X = df.drop('phishing', axis=1)   # features
    y = df['phishing']                # target
    
    # Split into train/test (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training set:", X_train.shape, y_train.shape)
    print("Testing set :", X_test.shape, y_test.shape)
Here you define predictors X and the binary label y (column phishing), then split the data into training and test sets. stratify=y keeps the class ratio consistent across splits (important for phishing datasets). random_state=42 makes the split reproducible. The shape prints confirm the sizes.

# Scale
    # ==========================
    # Feature Scaling
    # ==========================
    scaler = StandardScaler()
    
    # Fit on training data, transform both train & test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Data preparation completed!")
Standardization is learned only on training data (fit) and then applied to both sets (transform) to avoid data leakage. StandardScaler turns each numeric feature into z-scores: (x − mean_train) / std_train, which helps many models converge and prevents features with large ranges from dominating.

# Model Training
## Imports, load, and feature/label setup

    # =========================================
    # Neural Network (fixed batch size, clean logs)
    # =========================================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score,
        roc_curve, auc, precision_recall_curve, average_precision_score
    )
    
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    # ---- Load cleaned data ----
    FILE = '/content/drive/My Drive/Colab Notebooks/dataset_full_clean.csv'
    df = pd.read_csv(FILE)
    
    X = df.drop('phishing', axis=1).values
    y = df['phishing'].values.astype(int)
You import NumPy/pandas/matplotlib for data and plotting, scikit-learn for splitting, scaling, and metrics, and TensorFlow/Keras for the neural net. Then you load the cleaned CSV, split it into features X (all columns except phishing) and the binary target y (cast to int). Using the already-clean file avoids issues from NaNs/duplicates at the modeling stage.
