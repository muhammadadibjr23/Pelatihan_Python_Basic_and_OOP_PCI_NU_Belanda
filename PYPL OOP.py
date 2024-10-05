import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

class StockPricePredictor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = LinearRegression()
        
    def download_data(self):
        # Mengunduh data harga saham dari Yahoo Finance
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data.reset_index(inplace=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.data['Close'] = self.data['Close'].astype(float)

    def prepare_data(self):
        # Menyiapkan data untuk regresi
        self.data['Days'] = (self.data.index - self.data.index[0]).days  # Menghitung hari sejak awal
        X = self.data['Days'].values.reshape(-1, 1)  # Fitur
        y = self.data['Close'].values  # Target
        self.model.fit(X, y)
        self.data['Trend'] = self.model.predict(X)  # Interpolasi

    def predict_future(self, days_ahead=365):
        # Memprediksi satu tahun ke depan
        future_days = np.arange(self.data['Days'].max() + 1, 
                                self.data['Days'].max() + days_ahead + 1).reshape(-1, 1)
        future_prices = self.model.predict(future_days)

        # Membuat DataFrame untuk prediksi
        future_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
        predicted_prices = pd.DataFrame(data=future_prices, index=future_dates, columns=['Predicted'])
        return predicted_prices

    def plot_results(self, predicted_prices):
        # Menggabungkan data asli dengan prediksi
        full_data = pd.concat([self.data[['Close', 'Trend']], predicted_prices], axis=1)

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(full_data.index, full_data['Close'], label='Harga Saham Aktual', color='blue')
        plt.plot(full_data.index, full_data['Trend'], label='Interpolasi (Trend)', color='orange')
        plt.plot(predicted_prices.index, predicted_prices['Predicted'], label='Prediksi Harga 1 Tahun Kedepan', color='red')
        plt.title(f'Harga Saham {self.ticker} dan Prediksi 1 Tahun Kedepan')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga (USD)')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    predictor = StockPricePredictor(ticker='PYPL', start_date='2019-09-27', end_date='2024-09-27')
    predictor.download_data()
    predictor.prepare_data()
    predicted_prices = predictor.predict_future()
    predictor.plot_results(predicted_prices)
