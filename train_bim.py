import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

if not os.path.exists('data_bim.csv'):
    print("ERROR: Fail 'data_bim.csv' tak jumpa! Kena rakam data dulu guna kumpul_bim.py")
else:
    print("Membaca data...")
    data = pd.read_csv('data_bim.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    with open('model_bim.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Settle! Fail 'model_bim.p' telah dicipta.")