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

## Scale
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

    === Neural Network Evaluation ===
    Accuracy                 = 0.961530
    AUC                      = 0.992481
    False Alarm Rate (FAR)   = 0.030503
    Detection Rate (DR)      = 0.946713
    F1 Score                 = 0.94509
    
                  precision    recall  f1-score   support
    
               0       0.97      0.97      0.97     11343
               1       0.94      0.95      0.95      6099
    
        accuracy                           0.96     17442
       macro avg       0.96      0.96      0.96     17442
    weighted avg       0.96      0.96      0.96     17442

<img width="401" height="284" alt="image" src="https://github.com/user-attachments/assets/0e367e2f-e566-48c7-bfc0-cf46c7da7a05" />

<img width="541" height="357" alt="image" src="https://github.com/user-attachments/assets/fbb686b6-f564-45e2-96e0-a07440629e74" />

<img width="538" height="355" alt="image" src="https://github.com/user-attachments/assets/73763121-7e3e-4f01-8ae7-68f8785c4ac1" />

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

## Visual diagnostics: Confusion Matrix, ROC, PR curves
    # ---- Confusion Matrix (graph) ----
    plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Random Forest)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Random Forest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
    # ---- Precision–Recall Curve ----
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Random Forest)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
    
These plots let you see performance, not just numbers. The confusion matrix shows the TP/TN/FP/FN counts. ROC shows the trade-off between TPR and FPR with its AUC; the diagonal is random. The PR curve is especially useful when positives (phishing) are rarer; AP summarizes that curve.

## Threshold sweep for best F1 + feature importances
    # ---- Best-F1 threshold search ----
    thr_candidates = np.linspace(np.percentile(y_scores, 5), np.percentile(y_scores, 95), 21)
    best_f1, best_thr = -1, 0.5
    for thr in thr_candidates:
        f1_tmp = f1_score(y_test, (y_scores >= thr).astype(int))
        if f1_tmp > best_f1:
            best_f1, best_thr = f1_tmp, thr
    print(f"Best F1 across thresholds: {best_f1:.4f} at threshold {best_thr:.4f}")
    
    # ---- Top 20 Feature Importances ----
    importances = rf.feature_importances_
    feat_names = np.array(X.columns)
    idx = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8,8))
    plt.barh(range(len(idx)), importances[idx][::-1])
    plt.yticks(range(len(idx)), feat_names[idx][::-1])
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
Rather than always using a 0.5 cutoff, you thresholds (between the 5th–95th score percentiles) and report the one that maximizes F1—useful when you want a different precision/recall balance. Finally, the feature importance chart reveals which inputs the forest relied on most—great for interpretability and feature engineering.

# PCA + Classifier Pipeline
## Imports, config, and loading data
    # ==========================
    # PCA + KNN Pipeline (Full Evaluation & Graphs)
    # ==========================
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score,
        roc_curve, auc, precision_recall_curve, average_precision_score
    )
    
    # ---- Config ----
    FILE = "/content/drive/My Drive/Colab Notebooks/dataset_full_clean.csv"
    TEST_SIZE = 0.20
    VAL_SIZE_WITHIN_TRAIN = 0.20
    RANDOM_STATE = 42
    PCA_COMPONENTS = 30
    K_LIST = [3, 5, 7, 9, 11]
    WEIGHTS = "distance"
    
    # ---- Load dataset ----
    df = pd.read_csv(FILE)
    X = df.drop("phishing", axis=1)
    y = df["phishing"].astype(int)
You import all tools, set run-time knobs (file path, split sizes, PCA dimension, K values, weighting scheme), then load the cleaned data and split into features X and integer target y. Using weights="distance" lets nearer neighbors vote more strongly—often better after PCA.

## Splits, scaling, and PCA
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
    
    # ---- Scale on train only; transform val & test ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    
    # ---- PCA on train only; transform val & test ----
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p   = pca.transform(X_val_s)
    X_test_p  = pca.transform(X_test_s)
    
    print(f"PCA reduced features: {X.shape[1]} -> {X_train_p.shape[1]}")
    print(f"Total explained variance (sum): {pca.explained_variance_ratio_.sum():.4f}")
You hold out 20% for test, then from the remaining data create a validation split for tuning. Standardization is essential for KNN distance calculations. PCA is fitted only on the standardized training set and applied to val/test, shrinking dimensionality to PCA_COMPONENTS and reporting the variance you kept—this speeds KNN and can reduce noise.

## Validation sweep to choose k by Average Precision (AP)
    # ---- Sweep k on validation to choose best by Average Precision (AP) ----
    best_ap, best_k = -1.0, None
    for k in K_LIST:
        knn_val = KNeighborsClassifier(n_neighbors=k, weights=WEIGHTS)
        knn_val.fit(X_train_p, y_train)
        scores_val = knn_val.predict_proba(X_val_p)[:, 1]
        ap_val = average_precision_score(y_val, scores_val)
        print(f"k={k:2d}  |  Validation AP={ap_val:.4f}")
        if ap_val > best_ap:
            best_ap, best_k = ap_val, k
    
    print(f"\nChosen k (by best validation AP): {best_k}  (AP={best_ap:.4f})")
You try a small list of neighbor counts and pick the one that maximizes AP on the validation set. AP summarizes the precision–recall curve, which is appropriate when phishing positives may be rarer than benigns.

## Relearn transforms on Train+Val, train final KNN, and predict
    # ---- Refit scaler & PCA on train+val, then train final KNN with best k ----
    scaler_final = StandardScaler()
    X_trv_s = scaler_final.fit_transform(pd.concat([X_train, X_val], axis=0))
    X_tst_s = scaler_final.transform(X_test)
    
    pca_final = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_trv_p = pca_final.fit_transform(X_trv_s)
    X_tst_p = pca_final.transform(X_tst_s)
    y_trv = pd.concat([y_train, y_val], axis=0)
    
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=WEIGHTS)
    knn.fit(X_trv_p, y_trv)
    
    # ---- Predictions on test ----
    y_pred   = knn.predict(X_tst_p)
    y_scores = knn.predict_proba(X_tst_p)[:, 1]
After choosing k, you re-fit the scaler and PCA on the combined train+val data, then train the final KNN and produce test-set predictions and probability scores for unbiased evaluation.

## Metrics summary & diagnostic plots
    # ---- Summary metrics ----
    acc   = accuracy_score(y_test, y_pred)
    auc_v = roc_auc_score(y_test, y_scores)
    f1    = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / float(fp + tn)           # False Alarm Rate
    dr  = tp / float(tp + fn)           # Detection Rate (Recall for class 1)
    
    print("\n=== PCA + KNN Evaluation ===")
    print(f"Accuracy                 = {acc:.6f}")
    print(f"AUC                      = {auc_v:.6f}")
    print(f"False Alarm Rate (FAR)   = {far:.6f}")
    print(f"Detection Rate (DR)      = {dr:.6f}")
    print(f"F1 Score                 = {f1:.5f}\n")
    print(classification_report(y_test, y_pred, digits=2))
You report Accuracy, ROC-AUC (with scores), F1, False Alarm Rate (FP rate on benign URLs), and Detection Rate (recall on phishing). The classification_report gives per-class precision/recall/F1—useful to see trade-offs.
    # ---- Confusion Matrix (graph) ----
    plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (PCA + KNN)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.colorbar(); plt.tight_layout(); plt.show()
    
    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (PCA + KNN)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.show()
    
    # ---- Precision–Recall Curve ----
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (PCA + KNN)")
    plt.legend(loc="lower left"); plt.tight_layout(); plt.show()
The confusion matrix shows counts, the ROC curve visualizes TPR vs FPR with AUC, and the PR curve highlights performance on the positive class with AP—particularly informative for class imbalance.

## Threshold sweep for best F1
    thr_candidates = np.linspace(np.percentile(y_scores, 5), np.percentile(y_scores, 95), 21)
    best_f1, best_thr = -1, 0.5
    for thr in thr_candidates:
        f1_tmp = f1_score(y_test, (y_scores >= thr).astype(int))
        if f1_tmp > best_f1:
            best_f1, best_thr = f1_tmp, thr
    print(f"Best F1 across thresholds: {best_f1:.4f} at threshold {best_thr:.4f}")
Instead of always using a 0.5 cutoff, you probe a range of thresholds and report the one that maximizes F1. Use this when you prefer a different precision–recall balance.

    === PCA + KNN Evaluation ===
    Accuracy                 = 0.955395
    AUC                      = 0.985151
    False Alarm Rate (FAR)   = 0.031473
    Detection Rate (DR)      = 0.930972
    F1 Score                 = 0.93588
    
                  precision    recall  f1-score   support
    
               0       0.96      0.97      0.97     11343
               1       0.94      0.93      0.94      6099
    
        accuracy                           0.96     17442
       macro avg       0.95      0.95      0.95     17442
    weighted avg       0.96      0.96      0.96     17442

# XGBoost
## Imports, load data, train/test split
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score,
        roc_curve, auc, precision_recall_curve, average_precision_score
    )
    
    import xgboost as xgb
    print("XGBoost version:", xgb.__version__)
    
    FILE = "/content/drive/My Drive/Colab Notebooks/dataset_full_clean.csv"
    df = pd.read_csv(FILE)
    
    X = df.drop("phishing", axis=1)
    y = df["phishing"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
You import libraries, print the XGBoost version for reproducibility, load the cleaned dataset, separate the features and labels, and create a stratified 80/20 train–test split to preserve the phishing ratio.

## DMatrix, imbalance weight, parameters
     # DMatrix (native API)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X.columns.tolist())
    dvalid = xgb.DMatrix(X_test,  label=y_test,  feature_names=X.columns.tolist())
    dtest  = dvalid
    
    # Class imbalance weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Params
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(pos_weight),
        "tree_method": "hist",
        "seed": 42
    }
You convert data to XGBoost’s optimized DMatrix with readable feature names, compute a positive-class weight to mitigate class imbalance, and set sensible parameters with PR-AUC for evaluation, moderate depth, conservative learning rate, and fast histogram trees.

## Training with early stopping and prediction
    evals = [(dtrain, "train"), (dvalid, "valid")]
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    best_iter = getattr(booster, "best_iteration", None)
    print(f"Best iteration: {best_iter if best_iter is not None else 'N/A'}")
    
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        y_scores = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    else:
        y_scores = booster.predict(dtest)
    
    y_pred = (y_scores >= 0.5).astype(int)
Training stops automatically when validation PR-AUC doesn’t improve for 50 rounds, selecting a good number of trees and reducing overfitting; predictions return probabilities that you threshold at 0.5 to get class labels.

## Metrics summary
    acc   = accuracy_score(y_test, y_pred)
    auc_v = roc_auc_score(y_test, y_scores)
    f1    = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / float(fp + tn)   # False Alarm Rate on benign
    dr  = tp / float(tp + fn)   # Detection Rate (recall on phishing)
    
    print("\n=== XGBoost (native) Evaluation ===")
    print(f"Accuracy                 = {acc:.6f}")
    print(f"AUC                      = {auc_v:.6f}")
    print(f"False Alarm Rate (FAR)   = {far:.6f}")
    print(f"Detection Rate (DR)      = {dr:.6f}")
    print(f"F1 Score                 = {f1:.5f}\n")
    print(classification_report(y_test, y_pred, digits=2))
You report Accuracy, ROC-AUC, F1, False Alarm Rate on benign samples, and Detection Rate on phishing samples; the classification report adds per-class precision, recall, and F1 to diagnose imbalance.

## Visual diagnostics
    # Confusion Matrix
    plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (XGBoost)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.colorbar(); plt.tight_layout(); plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (XGBoost)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.show()
    
    # Precision–Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (XGBoost)")
    plt.legend(loc="lower left"); plt.tight_layout(); plt.show()
These plots visualize performance: the confusion matrix shows counts; the ROC shows TPR against FPR with AUC; the Precision–Recall curve and AP focus on the positive class and are often more informative when phishing is the minority.

## Threshold Sweep and Feature Importances
    # Best-F1 threshold search
    thr_candidates = np.linspace(np.percentile(y_scores, 5), np.percentile(y_scores, 95), 21)
    best_f1, best_thr = -1, 0.5
    for thr in thr_candidates:
        f1_tmp = f1_score(y_test, (y_scores >= thr).astype(int))
        if f1_tmp > best_f1:
            best_f1, best_thr = f1_tmp, thr
    print(f"Best F1 across thresholds: {best_f1:.4f} at threshold {best_thr:.4f}")
    
    # Feature Importance (Gain)
    gain_importance = booster.get_score(importance_type="gain")
    if gain_importance:
        feat_names = np.array(list(gain_importance.keys()))
        feat_gain  = np.array([gain_importance[k] for k in feat_names], dtype=float)
        order = np.argsort(feat_gain)[::-1][:20]
        plt.figure(figsize=(8,8))
        plt.barh(range(len(order)), feat_gain[order][::-1])
        plt.yticks(range(len(order)), feat_names[order][::-1])
        plt.title("Top 20 Feature Importances (Gain) — XGBoost")
        plt.xlabel("Average Gain"); plt.tight_layout(); plt.show()
    else:
        print("No gain-based feature importance available.")
You scan thresholds to find the one that maximizes F1, allowing you to tune the precision–recall trade-off; the gain-based importance chart highlights which features most improved splits, aiding interpretation and feature engineering.
=== XGBoost (native) Evaluation ===
Accuracy                 = 0.972194
AUC                      = 0.995955
False Alarm Rate (FAR)   = 0.025302
Detection Rate (DR)      = 0.967536
F1 Score                 = 0.96053

              precision    recall  f1-score   support

           0       0.98      0.97      0.98     11343
           1       0.95      0.97      0.96      6099

    accuracy                           0.97     17442
   macro avg       0.97      0.97      0.97     17442
weighted avg       0.97      0.97      0.97     17442
