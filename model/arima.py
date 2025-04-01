from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os

def model_arima(df, name, name_df):
    """
    Xây dựng mô hình ARIMA và lưu mô hình đã huấn luyện.
    
    Input:
    - df (DataFrame): Dữ liệu chuỗi thời gian.
    - name (str): Tên cột dữ liệu.
    - name_df (str): Tên DataFrame dùng trong tên tệp lưu mô hình.
    
    Output:
    - model (ARIMAResultsWrapper): Mô hình ARIMA, scaler đã huấn luyện.
    """

    # build model
    arima_model = ARIMA(df[name], order=(1,1,1))
    best_model = arima_model.fit()

    for i,j in zip(range(1,3), range(1,3)):
        arima_model = ARIMA(df[name], order=(1,i,j))
        model = arima_model.fit()
        if(np.abs(model.aic) < np.abs(best_model.aic)):
            best_model = model
    
    # save model
    os.makedirs(f'./model/{name_df}', exist_ok=True)
    with open(f'./model/{name_df}/arima_{name}.pkl', 'wb') as f:
        pickle.dump(best_model, f)


    return model

def save_predict_image(model, df, name, name_df, start_time, end_time):
    """
    Lưu hình ảnh dự đoán của mô hình ARIMA so với dữ liệu thực tế.
    
    Input:
    - model (ARIMAResultsWrapper): Mô hình ARIMA.
    - df (DataFrame): Dữ liệu chuỗi thời gian.
    - name (str): Tên cột dữ liệu.
    - name_df (str): Tên DataFrame dùng trong tên tệp lưu hình ảnh.
    - start_time (datetime): Thời gian bắt đầu dự đoán.
    - end_time (datetime): Thời gian kết thúc dự đoán.

    Output:
    - Lưu ảnh vào folder './image/{name}/'
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax = df.plot(ax=ax, label='Dữ liệu thực')
    plot_predict(model, start_time, end_time, ax=ax, dynamic=False, plot_insample=False)
    
    # Thêm nhãn và tiêu đề
    plt.xlabel('Thời gian')
    plt.ylabel('Giá trị')
    plt.title(f'Dự đoán dữ liệu từ {start_time} đến {end_time} \n ({name})', fontsize=14)
    plt.legend(['Dữ liệu thực', 'Dự đoán'], loc='upper left')
    
    plt.tight_layout()  # Điều chỉnh bố cục để tránh chồng chéo
    os.makedirs(f'./image/{name_df}', exist_ok=True)
    plt.savefig(f'./image/{name_df}/{name}_{start_time.strftime("%Y-%m")}_{end_time.strftime("%Y-%m")}.png')
    print(f"./image/{name_df}/{name}_{start_time.strftime('%Y-%m')}_{end_time.strftime('%Y-%m')}.png")
    plt.close('all') 
    # plt.show()

def save_predictions_to_csv(model, name, name_df, start_time, end_time):
    """
    Lưu tất cả các giá trị dự đoán của mô hình ARIMA vào một tệp CSV.
    
    Đầu vào:
    - model (ARIMAResultsWrapper): Mô hình ARIMA đã huấn luyện.
    - name (str): Tên cột dữ liệu.
    - name_df (str): Tên DataFrame dùng trong tên tệp lưu dữ liệu dự đoán.
    - start_time (datetime): Thời gian bắt đầu dự đoán.
    - end_time (datetime): Thời gian kết thúc dự đoán.

    Output:
    - Lưu kết quả vào folder './result/{name_df}/'
    """

    forecast = model.get_forecast(len(pd.date_range(start=start_time, end=end_time, freq="1MS")))
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    predictions = pd.DataFrame({
        'time': pd.date_range(start=start_time, end=end_time, freq="1MS"),
        'forecast_mean': forecast_mean,
        'lower_value': conf_int.iloc[:, 0],
        'upper_value': conf_int.iloc[:, 1],
    })

    os.makedirs(f'./result/{name_df}', exist_ok=True)
    predictions.to_csv(f'./result/{name_df}/{name}_predictions_{start_time.strftime("%Y-%m")}_{end_time.strftime("%Y-%m")}.csv', index=False)

if __name__ == '__main__':

    # Thời gian tới ngày train
    time_train = '2023-12-01'

    # Đọc dữ liệu
    # name_df = "SanXuat"
    # name_df = "TieuThu"
    name_df = "TonKho"
    
    df = pd.read_csv(f'../data/data_{name_df}_month.csv')

    # Chuyển đổi cột thời gian thành datetime
    df['time'] = pd.to_datetime(df['time'])

    # # Chạy, lưu model
    # for name in df.drop(columns=['time']):
    #     df_current = df[['time', name]]
    #     # Thiết lập cột thời gian làm chỉ số
    #     df_current.set_index('time', inplace=True)

    #     df_train = df_current[:time_train]

    #     model = model_arima(df_train, name, name_df)

    # # Thời gian dự đoán
    # start_time = pd.to_datetime('2024-01')
    # end_time = pd.to_datetime('2024-03')

    # # Lưu kết quả dự đoán
    # for name in df.drop(columns=['time']):
    #     df_current = df[['time', name]]
    #     # Thiết lập cột thời gian làm chỉ số
    #     df_current.set_index('time', inplace=True)

    #     df_train = df_current[:time_train]

    #     # Load the model using pickle
    #     with open(f'./model/{name_df}/arima_{name}.pkl', 'rb') as f:
    #         loaded_model = pickle.load(f)

    #     save_predictions_to_csv(loaded_model, name, name_df, start_time, end_time)

    # Thời gian dự đoán
    start_time = df['time'][0]
    # start_time = pd.to_datetime('2024-01')
    end_time = pd.to_datetime('2024-03')

    # Lưu hình ảnh kết quả dự đoán
    for name in df.drop(columns=['time']):
        df_current = df[['time', name]]
        # Thiết lập cột thời gian làm chỉ số
        df_current.set_index('time', inplace=True)

        df_train = df_current[:time_train]

        # Đọc model bằng pickle
        with open(f'./model/{name_df}/arima_{name}.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        print(name)

        save_predict_image(loaded_model, df_current, name, name_df, start_time, end_time)
