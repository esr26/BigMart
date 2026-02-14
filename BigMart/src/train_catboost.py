import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from preprocessing import preprocess


import os

def train_and_predict():


    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(BASE_DIR, "data", "train.csv")
    test_path = os.path.join(BASE_DIR, "data", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)



    train_df = preprocess(train_df).reset_index(drop=True)
    test_df = preprocess(test_df).reset_index(drop=True)

    X = train_df.drop('Item_Outlet_Sales', axis=1)
    y = train_df['Item_Outlet_Sales']

    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    test_preds = np.zeros(test_df.shape[0])
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            loss_function='RMSE',
            random_state=42,
            verbose=False,
            thread_count=-1
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=200
        )

        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_scores.append(rmse)

        print(f"Fold {fold} RMSE: {rmse:.4f}")

        test_preds += model.predict(test_df) / n_splits

    print("\nFinal CV RMSE:", np.mean(val_scores))

    return test_preds
