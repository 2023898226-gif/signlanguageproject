import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# 1. Load data dari CSV
print("Tengah baca data dari CSV... Sabar jap.")
data_df = pd.read_csv('data_asl.csv', header=None)

# 2. Asingkan data (X) dan label (y)
X = data_df.iloc[:, :-1].values # Semua koordinat
y = data_df.iloc[:, -1].values  # Huruf (Label)

# 3. Pecahkan data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 4. Bina Model Random Forest
print("Tengah training model... 10k data ni, bagi dia masa sikit.")
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# 5. Check Accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'Settle! Tahap ketepatan model: {score * 100:.2f}%')

# 6. Simpan model jadi fail 'model.p'
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

print("Fail 'model.p' dah siap simpan. Sekarang dah boleh run Streamlit!")