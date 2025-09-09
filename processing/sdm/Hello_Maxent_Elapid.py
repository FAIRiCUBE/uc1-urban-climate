import numpy as np
import pandas as pd
import elapid
from sklearn.metrics import roc_auc_score

"""
elapid Maxent model tutorial example
"""
if __name__ == "__main__":
    # Step 1: Generating synthetic data
    presence_data = pd.DataFrame(
        {
            "longitude": [10.0, 12.1, 13.5, 11.2, 14.3],
            "latitude": [60.1, 61.0, 59.9, 60.5, 61.2],
            "temperature": [15, 18, 16, 14, 19],
            "precipitation": [300, 400, 350, 450, 500],
        }
    )

    background_data = pd.DataFrame(
        {
            "longitude": np.random.uniform(10, 15, 100),
            "latitude": np.random.uniform(59, 62, 100),
            "temperature": np.random.uniform(10, 20, 100),
            "precipitation": np.random.uniform(200, 600, 100),
        }
    )

    # Step 2: Combine data and create labels
    presence_labels = np.ones(len(presence_data))
    background_labels = np.zeros(len(background_data))
    combined_data = pd.concat([presence_data, background_data], ignore_index=True)
    labels = np.concatenate([presence_labels, background_labels])

    # Features (environmental variables)
    features = combined_data[["temperature", "precipitation"]]

    # Step 3: Initialize and train MaxEnt model
    model = elapid.MaxentModel()
    model.fit(features, labels)

    # OPTIONAL but might be useful: Feature transformation.
    # model = elapid.MaxentModel(feature_types=["linear"])
    # featuresTransform = elapid.MaxentFeatureTransformer()
    # z = featuresTransform.fit_transform(features)
    # model.fit(z, labels)

    # Step 4: Make predictions for new environmental data
    new_data = pd.DataFrame(
        {"temperature": [17, 15, 12], "precipitation": [350, 420, 480]}
    )

    predictions = model.predict(new_data)
    print("Predicted probabilities:", predictions)

    # Step 5: Evaluate the model (optional)
    predicted_probabilities = model.predict(features)
    auc_score = roc_auc_score(labels, predicted_probabilities)
    print(f"Model AUC: {auc_score}")
