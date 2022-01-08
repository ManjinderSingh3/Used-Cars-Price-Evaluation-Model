from typing import Dict
from sklearn.model_selection import train_test_split
from file_processing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os.path
import time

before_label_encoding = read_csv_file(Path('Data', 'vehicle_final.csv'))
after_label_encoding = read_csv_file(Path('Data', 'vehicle_final_le.csv'))
rf_F_name = "Random_Forest_Model.pkl"

# MODEL:
def random_forest_regressor(x: pd.DataFrame, y: pd.Series) -> Dict:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x_train, y_train)
    predictions = rf_regressor.predict(x_test)
    accuracy = round(r2_score(y_test, predictions) * 100, 2)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rf_model = rf_regressor
    rf_acc = accuracy
    rf_mse = mse
    rf_mae = mae
    # Model will be saved during the first iteration only.
    file_name = os.path.isfile(rf_F_name)

    # If File does not exist than this section of saving model will be executed. It will be executed only once.
    if (file_name == 0):
        save_random_forest_regressor(rf_regressor)
    return dict(model=rf_model, r2_score=rf_acc, mse=rf_mse, mae=rf_mae)


def save_random_forest_regressor(rf_model):
    pickle.dump(rf_model, open('Random_Forest_Model.pkl', 'wb'))


def run_regression_models(x, y):
    r1 = random_forest_regressor(x, y)
    return r1

if __name__=="__main__":
    vehicles_df = pd.read_csv(Path('Data', 'vehicle_final_le.csv'))
    start = time.time()
    print("Regression in progress...")
    output_col = "price"
    feature_cols = vehicles_df.columns.tolist()
    feature_cols.remove(output_col)
    x_features = vehicles_df[feature_cols]
    y_label = vehicles_df[output_col]

    result = run_regression_models(x_features, y_label)
    print(f"{10 * '*'}Dataset1{vehicles_df.shape}, usecase1: {10 * '*'}\nRandom Forest Regressor: {result}\n")

    end = time.time()
    run_time = round(end - start, 4)
    print("Regression ended...")
    print(f"{30 * '-'}\nRegression run_time:{run_time}s\n{30 * '-'}\n")



