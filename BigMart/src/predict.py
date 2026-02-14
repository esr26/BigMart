import pandas as pd
import os
from train_catboost import train_and_predict


def generate_submission():

    # Get base directory (project root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths
    test_path = os.path.join(BASE_DIR, "data", "test.csv")
    output_path = os.path.join(BASE_DIR, "outputs", "final_submission.csv")

    # Generate predictions
    test_preds = train_and_predict()

    # Load original test (for IDs only)
    original_test = pd.read_csv(test_path)

    # Create submission
    submission = pd.DataFrame({
        'Item_Identifier': original_test['Item_Identifier'],
        'Outlet_Identifier': original_test['Outlet_Identifier'],
        'Item_Outlet_Sales': test_preds
    })

    submission.to_csv(output_path, index=False)

    print("\nSubmission file created successfully!")
    print("Saved at:", output_path)


if __name__ == "__main__":
    generate_submission()
