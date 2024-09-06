# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Membaca data historis emas
data = pd.read_csv('Data Historis GLD (1).csv', parse_dates=['Tanggal'], dayfirst=True)

# Mengganti tanda koma dengan titik dalam kolom yang berisi nilai numerik
numeric_cols = ['Pembukaan', 'Tertinggi', 'Terendah', 'Terakhir']
for col in numeric_cols:
    data[col] = data[col].str.replace(',', '.').astype(float)

# Mengganti tanda koma dengan titik dalam kolom Vol.
data['Vol.'] = data['Vol.'].str.replace(',', '').str.replace('M', '').astype(float) * 1e6

# Mengonversi kolom Change% menjadi desimal
data['Perubahan%'] = data['Perubahan%'].str.replace(',', '.').str.replace('%', '').astype(float) / 100

# Persiapan data
data['Tanggal'] = pd.to_datetime(data['Tanggal'])
data.set_index('Tanggal', inplace=True)

df_unsampled = data.resample('D').asfreq().interpolate(method='linear')
df_unsampled.reset_index(inplace=True)
data = data.sort_values('Tanggal')
df_unsampled = df_unsampled.sort_values('Tanggal')

import seaborn as sns
import matplotlib.pyplot as plt

# Pilih subset kolom yang relevan
selected_columns = ['Pembukaan', 'Tertinggi', 'Terendah', 'Terakhir', 'Vol.', 'Perubahan%']
data_subset = df_unsampled[selected_columns]

# Hitung korelasi hanya terhadap kolom 'Terakhir'
corr_with_close = data_subset.corr(method='pearson')[['Terakhir']].drop('Terakhir')

# Buat heatmap untuk korelasi dengan 'Terakhir'
plt.figure(figsize=(5, 7))
sns.heatmap(corr_with_close, annot=True, annot_kws={"size": 10}, cmap='coolwarm', cbar=True, fmt='.4f')

plt.title('Heatmap Korelasi dengan Terakhir (Close)')
plt.xlabel('Terakhir (Close)')
plt.ylabel('Variabel')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Pilih subset kolom yang relevan
selected_columns = ['Pembukaan', 'Tertinggi', 'Terendah', 'Terakhir', 'Vol.', 'Perubahan%']
data_subset = df_unsampled[selected_columns]

# Buat heatmap korelasi
corr_matrix = data_subset.corr(method='pearson')

# Buat heatmap dengan anotasi manual
sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 10}, cmap='coolwarm', cbar=True,
            xticklabels=selected_columns, yticklabels=selected_columns, fmt='')

# Atur anotasi secara manual agar tidak dibulatkan
for text in plt.gca().texts:
    text.set_text(f'{float(text.get_text()):.4f}')  # Menampilkan dengan 4 tempat desimal atau sesuai kebutuhan

plt.title('Heatmap Korelasi')
plt.show()

# Pilih fitur-fitur yang akan digunakan untuk memprediksi harga emas
features = ['Pembukaan', 'Tertinggi', 'Terendah']
target = ['Terakhir']
X = df_unsampled[features]
y = df_unsampled[target].shift(-1).dropna()
X = X.iloc[:-1,:]

print(df_unsampled)
print(X)
print(y)

X_train = X.copy()
y_train = y.copy()

X_test = X.copy()
y_test = y.copy()

# Bagi data menjadi data pelatihan dan data pengujian
X_train = X_train.drop(X_train[2993:].index)
y_train = y_train.drop(y_train[2993:].index)
y_train.shape
X_train.shape

X_test = X_test.drop(X_test[:-749].index)
y_test = y_test.drop(y_test[:-749].index)
y_test.shape
X_test.shape

X_train2 = X_train.copy()
y_train2 = y_train.copy()

X_test2 = X_test.copy()
y_test2 = y_test.copy()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train2)
X_test_scaled = scaler.transform(X_test2)

# # Hitung kernel RBF antara setiap pasangan data
# def rbf_kernel(X_train_scaled):
#     num_data = len(X_train_scaled)
#     kernel_matrix = np.zeros((num_data, num_data))
#     gamma = 1
#     for i in range(num_data):
#         for j in range(num_data):
#             kernel_matrix[i, j] = np.exp(-gamma * np.sum((X_train_scaled[i] - X_train_scaled[j]) ** 2))
#     return kernel_matrix

# # Cetak matriks kernel
# kernel_matrix = rbf_kernel(X_train_scaled)
# print("Kernel Matrix:")
# print(kernel_matrix)

# kernel_df = pd.DataFrame(kernel_matrix, columns=range(1, len(kernel_matrix)+1), index=range(1, len(kernel_matrix)+1))
# # print(kernel_df).head()

# # Menyimpan DataFrame ke dalam file CSV
# kernel_df.to_csv('kernel_matrix.csv')

# # Lakukan split lagi pada data training untuk mendapatkan data validasi
# X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train2, y_train2, test_size=0.2, random_state=42)

# # Inisialisasi model SVR dengan kernel RBF
# model = SVR(kernel='rbf')

# # Latih model dengan data training
# model.fit(X_train_split, y_train_split)

# # Prediksi pada data validasi
# val_predictions = model.predict(X_val_split)

# Latih model Support Vector Regression (SVR)
model = SVR(kernel='rbf')  # Gunakan kernel radial basis function (RBF)
model.fit(X_train_scaled, y_train2)

# Prediksi harga emas pada data pengujian
predictions = model.predict(X_test_scaled)

#Hitung koefisien determinasi (R-squared)
r_squared = r2_score(y_test2, predictions)
print("(R-squared):", r_squared)
mse = mean_squared_error(y_test2, predictions)
print("mse",mse)
mape = mean_absolute_percentage_error(y_test2, model.predict(X_test_scaled))
print("mape",mape)

predictions = pd.DataFrame(predictions, columns=['Prediksi Harga'])
print(predictions)

df_pred = df_unsampled.drop(['Terakhir','Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%'], axis=1)
df_pred = df_pred.iloc[1:]
df_pred = df_pred.reset_index(drop=True)
df_pred = df_pred.drop(df_pred[:-521].index)
df_pred = df_pred['Tanggal'].reset_index()
preds = pd.concat([df_pred,predictions], axis=1)
preds = preds.drop(['index'], axis=1)

plt.figure(figsize=(14, 7))
plt.plot(pd.to_datetime(df_unsampled['Tanggal']), df_unsampled['Terakhir'], label='Nilai Aktual')
plt.plot(pd.to_datetime(preds['Tanggal']),preds['Prediksi Harga'], label='Hasil Prediksi', linestyle='--') # Use 'Prediksi Harga' instead of 'y_pred'
plt.xlabel('Tanggal')
plt.ylabel('Harga Terakhir')
plt.title('Perbandingan Nilai Aktual dan Hasil Prediksi Harga Emas')
plt.legend()
plt.grid(True)
plt.show()

# Input data baru
input_data = (189.68, 191.26, 189.08)

# Mengonversi input data menjadi array numpy
input_data_as_numpy_array = np.asarray(input_data)

# Melakukan reshape data input untuk memenuhi format yang diperlukan oleh model
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Melakukan scaling pada input data
input_data_scaled = scaler.transform(input_data_reshaped)

# Melakukan prediksi menggunakan model yang telah dilatih
prediction = model.predict(input_data_scaled)[0]
print("Prediksi harga emas untuk data input baru:", prediction)

# Prediksi harga emas untuk semua data
X_new_scaled = scaler.transform(X)  # Scale X using the same scaler
y_pred_all = model.predict(X_new_scaled)

# Menambahkan kolom prediksi ke DataFrame
df_unsampled['Terakhir_pred'] = np.nan
df_unsampled.iloc[:-1, df_unsampled.columns.get_loc('Terakhir_pred')] = y_pred_all  # Ensure lengths match by using iloc to slice the DataFrame before the new column is added

y_pred_all = pd.DataFrame(y_pred_all, columns=['Prediksi Harga'])
print(y_pred_all)

print(X_new_scaled)

# metrics = ['MAPE', 'R-squared', 'MSE']
# scores = [mape, r_squared, mse]

# # Plot diagram batang
# plt.figure(figsize=(10, 6))
# bars = plt.bar(metrics, scores, color=['blue', 'green', 'orange'])

# # Menambahkan label dan nilai di atas setiap bar
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')

# # Menambahkan judul dan label sumbu
# plt.title('Perbandingan Metrik Evaluasi')
# plt.xlabel('Metrik')
# plt.ylabel('Nilai')

# # Menampilkan plot
# plt.show()

print(data)

print(df_unsampled)

# Menyimpan DataFrame yang diperbarui ke file CSV
df_unsampled.to_csv('Hasil Model 1 dengan Prediksi.csv', index=False)

# import pickle
# model_data = {
#     'model': model,
#     'scaler': scaler,
#     'X_train_scaled': X_train_scaled
# }

# with open('model2.pkl', 'wb') as model_file:
#     pickle.dump(model_data, model_file)