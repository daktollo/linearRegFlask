import pandas as pd
import matplotlib.pyplot as plt
import joblib

# CSV dosyasını yükle
data = pd.read_csv('SOCR-HeightWeight.csv')

# Bağımsız ve bağımlı değişkenleri seç
X = data[['Height(Inches)']]
y = data['Weight(Pounds)']

# İlk olarak orijinal verileri görselleştirelim
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Orijinal Veriler')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.title('Orijinal Veriler')
plt.legend()
plt.show()

# Daha sonra kaydedilen modeli yükleyelim ve tahmin yapalım
model = joblib.load('linear_regression_model.pkl')

# Tahminler
y_pred = model.predict(X)

# Orijinal veriler ve model tahminlerini aynı grafikte görselleştirelim
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Orijinal Veriler')
plt.plot(X, y_pred, color='red', label='Model Tahminleri')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.title('Orijinal Veriler ve Model Tahminleri')
plt.legend()
plt.show()
