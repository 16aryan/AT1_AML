import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from pathlib import Path
from loguru import logger
from AT1_AML.config import PROCESSED_DATA_DIR, MODELS_DIR, PROJ_ROOT

def train_lgb():
    logger.info("Loading processed datasets...")
    train_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    
    if not train_path.exists() or not test_path.exists():
        logger.error("Processed datasets not found. Please run the preprocessing script first.")
        return
        
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # Identify feature columns
    drop_cols = ["pid", "label", "ht", "yr", "dob", "type", "num"]
    feature_cols = [col for col in train.columns if col not in drop_cols]
    
    logger.info(f"Using {len(feature_cols)} features for model training.")
    
    X = train[feature_cols]
    y = train["label"]
    X_test = test[feature_cols]
    
    # Configure categorical features for LightGBM
    categorical_cols = ["team", "conf", "role"]
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
            X_test[col] = X_test[col].astype("category")
            
    # CV setup
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    fold_scores = []
    feature_importances = np.zeros(len(feature_cols))
    
    logger.info("Starting 5-Fold Stratified Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Baseline-style high performing hyperparameters
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            class_weight="balanced",
            random_state=42 + fold,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
        )
        
        # Validation predictions
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        # Score fold AUPRC
        fold_auprc = average_precision_score(y_val, val_preds)
        fold_scores.append(fold_auprc)
        logger.info(f"Fold {fold + 1} AUPRC: {fold_auprc:.5f}")
        
        # Accumulate feature importances
        feature_importances += model.feature_importances_ / kf.n_splits
        
        # Accumulate test predictions
        test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits
        
    overall_auprc = average_precision_score(y, oof_preds)
    logger.info("--------------------------------------")
    logger.success(f"Mean Fold AUPRC: {np.mean(fold_scores):.5f}")
    logger.success(f"Overall OOF AUPRC: {overall_auprc:.5f}")
    logger.info("--------------------------------------")
    
    # Save feature importances
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importances
    }).sort_values(by="importance", ascending=False)
    
    logger.info("Top 15 Features by Importance:")
    for idx, row in fi_df.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.1f}")
        
    # Save the model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "lgb_model.txt"
    # We will fit a final model on 100% of data to save, or use the ensemble.
    # To save a representative model, we train one final model on the whole train set.
    logger.info("Training final model on full dataset...")
    final_model = lgb.LGBMClassifier(
        n_estimators=350,  # approximate early stopping round
        learning_rate=0.03,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    final_model.fit(X, y)
    final_model.booster_.save_model(str(best_model_path))
    logger.success(f"Final model saved to {best_model_path}")
    
    # Save submission file
    submission_path = PROJ_ROOT / "submission.csv"
    submission = pd.DataFrame({
        "pid": test["pid"],
        "label": test_preds
    })
    submission.to_csv(submission_path, index=False)
    logger.success(f"Kaggle submission file saved to {submission_path}")

if __name__ == "__main__":
    train_lgb()
