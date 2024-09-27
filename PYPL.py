import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Mengunduh data harga saham PYPL dari Yahoo Finance
ticker = 'PYPL'
data = yf.download(ticker, start='2019-09-27', end='2024-09-27')
data.reset_index(inplace=True)

# Memilih kolom tanggal dan harga penutupan
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['Close'] = data['Close'].astype(float)

# Menyiapkan data untuk regresi
data['Days'] = (data.index - data.index[0]).days  # Menghitung hari sejak awal
X = data['Days'].values.reshape(-1, 1)  # Fitur
y = data['Close'].values  # Target

# Membuat model regresi linear
model = LinearRegression()
model.fit(X, y)

# Melakukan interpolasi
data['Trend'] = model.predict(X)

# Memprediksi satu tahun ke depan
future_days = np.arange(data['Days'].max() + 1, data['Days'].max() + 365 + 1).reshape(-1, 1)
future_prices = model.predict(future_days)

# Membuat DataFrame untuk prediksi
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=365)
predicted_prices = pd.DataFrame(data=future_prices, index=future_dates, columns=['Predicted'])

# Menggabungkan data asli dengan prediksi
full_data = pd.concat([data[['Close', 'Trend']], predicted_prices], axis=1)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(full_data.index, full_data['Close'], label='Harga Saham Aktual', color='blue')
plt.plot(full_data.index, full_data['Trend'], label='Interpolasi (Trend)', color='orange')
plt.plot(predicted_prices.index, predicted_prices['Predicted'], label='Prediksi Harga 1 Tahun Kedepan', color='red')
plt.title('Harga Saham PayPal (PYPL) dan Prediksi 1 Tahun Kedepan')
plt.xlabel('Tanggal')
plt.ylabel('Harga (USD)')
plt.legend()
plt.grid()
plt.show()
