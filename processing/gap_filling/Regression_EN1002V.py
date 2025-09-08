from src.measurer import Measurer
from types import ModuleType
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import psutil

"""
Example of regression model for gap filling of EN1002V (average number of hours worked per day by employed persons) 
using EN1003V and EN1004V as predictors.
"""


def Regression_train(
    data_path="./data/test/UrbanAudit_ESTAT_EN1002V_EN1003V_EN1004V_long.csv",
    out_dir="./data/test/results",
    target="EN1002V",
):
    # Load your data from a CSV file
    data = pd.read_csv(data_path)
    # Update outliers (5 cities have reported total hours/years and not the daily average)
    data.loc[data["EN1002V"] > 24, "EN1002V"] = (
        data.loc[data["EN1002V"] > 24, "EN1002V"] / 367
    )

    corr_matrix = data.iloc[:, 2:].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    if out_dir != "":
        plt.savefig(out_dir + "/EN1002V_EN1003V_EN1004V_Correlation.pdf")

    X = data.iloc[:, 3:]
    y = data.iloc[:, 2]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor(n_estimators=125, random_state=42)

    # Train the regressor
    gb_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = gb_regressor.predict(X_test)

    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {abs(y_pred - y_test).sum() / len(y_pred)}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {math.sqrt(mse)}")
    print(f"R2: {r2}")

    joblib.dump(gb_regressor, out_dir + "/" + target + ".pkl")

    plt.hist(y_test, bins=5, color="blue", edgecolor="black", alpha=1, label="Y_test")
    plt.hist(y_pred, bins=5, color="red", edgecolor="black", alpha=0.8, label="Y_pred")

    plt.xlabel("Hours")
    plt.ylabel("Frequency")
    plt.title("Distribution of Data")
    plt.legend()
    if out_dir != "":
        plt.savefig(out_dir + "/EN1002V_EN1003V_EN1004V_Distribution.pdf")

    plt.plot(list(y_test), marker=".", linestyle="-", label="Y_test")
    plt.plot(list(y_pred), marker=".", linestyle="-", label="Y_pred")

    plt.xlabel("ID")
    plt.ylabel("Hours")
    plt.title("Y_test vs Y_predict")
    plt.legend()
    if out_dir != "":
        plt.savefig(out_dir + "/EN1002V_preditiction.pdf")
    print(
        f"Memory used for training the model: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)} MB"
    )
    print("Training completed.")


def Regression_predict(
    data_path="./data/test/UrbanAudit_ESTAT_missingEN1002V.csv",
    model_path="./data/test/results/EN1002V.pkl",
    out_dir="./data/test/results",
    target="EN1002V",
):
    # Load the saved model
    loaded_model = joblib.load(model_path)
    X_missing = pd.read_csv(data_path)
    X_missing[target] = loaded_model.predict(X_missing.iloc[:, 2:])
    if out_dir != "":
        X_missing.to_csv(out_dir + "/Filled_" + target + ".csv", index=False)
    print(
        f"Memory used for predicting the missing values: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)} MB"
    )
    print("Prediction completed.")


def main():
    # Path where the data are stored (the use of the disk in this path is measured).
    # Use '/' to measure the entire disk.
    data_path = "/"
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    # example -> shape = [5490, 2170]
    shape = []

    # Start
    Regression_train(
        data_path="./data/test/UrbanAudit_ESTAT_EN1002V_EN1003V_EN1004V_long.csv",
        out_dir="./data/test/results",
        target="EN1002V",
    )

    # End
    Regression_predict(
        data_path="./data/test/UrbanAudit_ESTAT_missingEN1002V.csv",
        model_path="./data/test/results/EN1002V.pkl",
        out_dir="./data/test/results",
        target="EN1002V",
    )

    # it is very important to use program_path = __file__
    measurer.end(
        tracker=tracker,
        shape=shape,
        libraries=[
            v.__name__
            for k, v in globals().items()
            if type(v) is ModuleType and not k.startswith("__")
        ],
        data_path=data_path,
        program_path=__file__,
        variables=locals(),
        csv_file=f"./data/test/results/EN1002V_benchmarks{datetime.now()}.csv",
    )


if __name__ == "__main__":
    main()
