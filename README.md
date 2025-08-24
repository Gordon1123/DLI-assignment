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

## Data Cleaning
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

## Train/val/test split and scaling 

    # ---- Split: train/val/test = 64% / 16% / 20% ----
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.20, stratify=y_tmp, random_state=42
    )
    
    # ---- Scale ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)


You first hold out 20% as a test set, then split the remaining 80% into train/validation (80/20 of that), yielding 64% train / 16% val / 20% test. stratify preserves the phishing/non-phishing ratio in every split. StandardScaler is fitted only on the training data to prevent leakage, and then applied to validation and test.

## Model architecture & compilation
    # ---- Model ----
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_s.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
This is a simple feed-forward network: two ReLU hidden layers (128 → 64), batch-norm after the first, and dropout (0.2) for regularization. The final sigmoid outputs a probability for the positive class. You compile with Adam (1e-3), binary cross-entropy, and track accuracy plus ROC-AUC.

## Callbacks and training loop
    # ---- Callbacks ----
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', mode='max', factor=0.5, patience=4, verbose=1
        )
    ]
    
    # ---- Train ----
    history = model.fit(
        X_train_s, y_train,
        epochs=50,
        batch_size=64,              # fixed; change if you like
        validation_data=(X_val_s, y_val),
        verbose=1,
        callbacks=cbs
    )
Early stopping watches validation AUC and restores the best weights to avoid overfitting; ReduceLROnPlateau halves the learning rate if val-AUC stalls.It train up to 50 epochs with a fixed batch size of 64, using the validation split for on-the-fly feedback.

## Evaluation metrics and summary printout
    # ---- Evaluate on test ----
    y_scores = model.predict(X_test_s).ravel()
    y_pred   = (y_scores >= 0.5).astype(int)
    
    acc   = accuracy_score(y_test, y_pred)
    auc_v = roc_auc_score(y_test, y_scores)
    f1    = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / float(fp + tn)     # False Alarm Rate
    dr  = tp / float(tp + fn)     # Detection Rate (recall for class 1)
    
    print("\n=== Neural Network Evaluation ===")
    print(f"Accuracy                 = {acc:.6f}")
    print(f"AUC                      = {auc_v:.6f}")
    print(f"False Alarm Rate (FAR)   = {far:.6f}")
    print(f"Detection Rate (DR)      = {dr:.6f}")
    print(f"F1 Score                 = {f1:.5f}\n")
    print(classification_report(y_test, y_pred, digits=2))
Predicted scores (probabilities) are thresholded at 0.5 to get class labels. You compute Accuracy, ROC-AUC (using scores), F1, the confusion matrix, False Alarm Rate (FP rate on negatives), and Detection Rate (recall on positives). classification_report prints precision/recall/F1 per class—useful if the dataset is imbalanced.

## Visualization: confusion matrix, ROC, and PR curves
    # ---- Confusion Matrix ----
    plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (NN)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.colorbar(); plt.tight_layout(); plt.show()
    
    # ---- ROC ----
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (NN)"); plt.legend(); plt.tight_layout(); plt.show()
    
    # ---- Precision–Recall ----
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (NN)"); plt.legend(); plt.tight_layout(); plt.show()
The confusion-matrix heatmap shows counts of TP/TN/FP/FN. The ROC curve plots TPR vs FPR with its AUC; the diagonal is random guessing. The Precision–Recall curve is especially informative for class imbalance; AP (Average Precision) summarizes area under the PR curve.

# Random Forest Algorithm (RF)
## Imports, config, and loading data

    # ==========================
    # Random Forest (RF) — Full Pipeline (Val sweep + Graphs)
    # ==========================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score,
        roc_curve, auc, precision_recall_curve, average_precision_score
    )
    
    # ---- Config ----
    FILE = "/content/drive/My Drive/Colab Notebooks/dataset_full_clean.csv"
    TEST_SIZE = 0.20
    VAL_SIZE_WITHIN_TRAIN = 0.20     # 20% of the train portion becomes validation
    RANDOM_STATE = 42
    CLASS_WEIGHT = "balanced"        # helps with class imbalance
    
    # ---- Load ----
    df = pd.read_csv(FILE)
    X = df.drop("phishing", axis=1)
    y = df["phishing"].astype(int)
You bring in NumPy/pandas/matplotlib, scikit-learn’s RF and metrics, and set run-time knobs (file path, split sizes, seed, and class_weight='balanced' to reduce class-imbalance bias). Then you load the cleaned dataset, split it into features X and integer target y (the phishing column).

## Train/test split + internal validation split
    # ---- Split into train/test ----
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # ---- Split train into train/val ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        stratify=y_train_full,
        random_state=RANDOM_STATE
    )
First you hold out a test set of 20% with stratification to preserve the class ratio. From the remaining training pool you carve out a validation set (20% of train), again stratified. That gives ~64% train, 16% validation, 20% test—clean separation for tuning (val) and final reporting (test).
## Small validation sweep

    # ---- Small validation sweep (optimize Average Precision on val) ----
    param_grid = {
        "n_estimators":     [200, 400, 600],
        "max_depth":        [None, 12, 20],
        "min_samples_leaf": [1, 3, 5],
    }
    
    best = (-1.0, None, None)  # (AP, params, model)
    for n in param_grid["n_estimators"]:
        for d in param_grid["max_depth"]:
            for m in param_grid["min_samples_leaf"]:
                rf_val = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=d,
                    min_samples_leaf=m,
                    class_weight=CLASS_WEIGHT,
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
                rf_val.fit(X_train, y_train)
                scores_val = rf_val.predict_proba(X_val)[:, 1]
                ap_val = average_precision_score(y_val, scores_val)
                print(f"n_estimators={n:4d} | max_depth={str(d):>4} | min_samples_leaf={m} -> Val AP={ap_val:.4f}")
                if ap_val > best[0]:
                    best = (ap_val, {"n_estimators": n, "max_depth": d, "min_samples_leaf": m}, rf_val)
    
    best_ap, best_params, _ = best
    print(f"\nChosen params (by best validation AP={best_ap:.4f}): {best_params}")
    
You loop over a compact grid of RF hyperparameters and train on train, score on validation, and keep the combination that maximizes Average Precision (AP)—a PR-curve metric well-suited to imbalanced phishing data. This lightweight sweep avoids full grid-search overhead but still finds a good configuration.

## Refit with the best params on train+val and predict on test
    # ---- Refit on Train+Val with best params ----
    X_trv = pd.concat([X_train, X_val], axis=0)
    y_trv = pd.concat([y_train, y_val], axis=0)
    
    rf = RandomForestClassifier(
        **best_params,
        class_weight=CLASS_WEIGHT,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_trv, y_trv)
    
    # ---- Test predictions ----
    y_pred   = rf.predict(X_test)
    y_scores = rf.predict_proba(X_test)[:, 1]
After choosing hyperparameters, you retrain on the combined train+val set to give the model more data. Then you produce class labels (predict) and probability scores (predict_proba) on the untouched test set for unbiased final evaluation.

## Metrics summary and report
    # ---- Summary metrics ----
    acc   = accuracy_score(y_test, y_pred)
    auc_v = roc_auc_score(y_test, y_scores)
    f1    = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / float(fp + tn)           # False Alarm Rate
    dr  = tp / float(tp + fn)           # Detection Rate (Recall for class 1)
    
    print("\n=== Random Forest Evaluation ===")
    print(f"Accuracy                 = {acc:.6f}")
    print(f"AUC                      = {auc_v:.6f}")
    print(f"False Alarm Rate (FAR)   = {far:.6f}")
    print(f"Detection Rate (DR)      = {dr:.6f}")
    print(f"F1 Score                 = {f1:.5f}\n")
    print(classification_report(y_test, y_pred, digits=2))
You compute key metrics: Accuracy, ROC-AUC (using scores), F1, plus False Alarm Rate (FP rate on benign sites) and Detection Rate (recall on phishing). The classification_report prints precision/recall/F1 per class—handy for imbalance diagnostics.
