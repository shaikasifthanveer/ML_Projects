import sqlite3
import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import yfinance as yf
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)
app.secret_key = 'infosys_springboard_secret_key'

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE NOT NULL, 
                  email TEXT NOT NULL,
                  password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# --- ROUTES ---

@app.route('/')
def home():
    """Serves the main landing page."""
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid Username or Password', 'error')
            
    return render_template('login.html', register_mode=False)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration Successful! Please Login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'error')
            
    return render_template('login.html', register_mode=True)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

# --- HIGH ACCURACY AI ENGINE ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.json
        symbol = data['symbol']
        seq_length = int(data['seq_length'])
        epochs = int(data['epochs'])

        # 1. Download Data
        print(f"Downloading high-accuracy data for {symbol}...")
        df = yf.download(symbol, period='max', interval='1d')
        df.dropna(inplace=True)
        
        if len(df) < seq_length + 50:
            return jsonify({'error': f'Not enough data for {symbol}'}), 400
            
        feature_cols = ['Open', 'High', 'Low', 'Close']
        features = df[feature_cols].values

        # 2. OPTIMIZATION: Limit to last 2000 points
        if len(features) > 2000:
            features = features[-2000:]
            df = df.iloc[-2000:]
            
        # 3. Preprocessing
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        X, y = [], []
        for i in range(seq_length, len(scaled_features)):
            X.append(scaled_features[i-seq_length:i])
            y.append(scaled_features[i, 3]) 

        X, y = np.array(X), np.array(y)

        # 80/20 Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        dates = df.index.strftime('%Y-%m-%d').tolist()
        test_dates = dates[seq_length:][split:]

        # 4. IMPROVED MODEL ARCHITECTURE
        model = Sequential([
            LSTM(80, return_sequences=True, input_shape=(seq_length, 4)),
            Dropout(0.2),
            LSTM(40),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # 5. EARLY STOPPING
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        # Train
        model.fit(X_train, y_train, 
                  epochs=epochs, 
                  batch_size=32, 
                  validation_split=0.1,
                  callbacks=[early_stop], 
                  verbose=0)

        # 6. Predict and Inverse Transform
        y_pred_scaled = model.predict(X_test)
        
        dummy_zeros = np.zeros((len(y_test), 3))
        y_true = scaler.inverse_transform(np.hstack([dummy_zeros, y_test.reshape(-1, 1)]))[:, 3]
        y_pred = scaler.inverse_transform(np.hstack([dummy_zeros, y_pred_scaled]))[:, 3]

        # Metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        latest_prediction = y_pred[-1]

        response = {
            'dates': test_dates,
            'actual': y_true.tolist(),
            'predicted': y_pred.tolist(),
            'symbol': symbol,
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'latest_price': round(latest_prediction, 2)
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)