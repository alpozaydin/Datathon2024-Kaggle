import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder

# Read the dataframe
df = pd.read_csv('/Users/alpozaydin/btk_akademi/model/data/2022.csv', encoding='utf-8')
#df2020 = pd.read_csv('data/2020.csv', encoding='utf-8')


#print(df.info())

# List of columns to be dropped
columns_to_be_dropped = ['Lise Adi', 'Burs Aldigi Baska Kurum', 'Hangi STK\'nin Uyesisiniz?',
                        'Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?', 'id',
                        'Dogum Yeri', 'Lise Sehir', 'Baba Calisma Durumu', 'Anne Calisma Durumu', 'Lise Turu', 'Dogum Tarihi']

# Use text fetching function from trash.py
import burs_kurumu_cleaning
df = burs_kurumu_cleaning.standardize_turkish_scholarship(df)

# Apply drop
df = df.drop(columns=columns_to_be_dropped)

# List of columns to be encoded
columns_to_encode = [
    'Cinsiyet', 'Ikametgah Sehri', 'Universite Adi', 'Universite Turu',
    'Bölüm', 'Universite Kacinci Sinif',
    'Lise Bolumu', 'Lise Mezuniyet Notu', 'Baska Bir Kurumdan Burs Aliyor mu?',
    'Baska Kurumdan Aldigi Burs Miktari', 'Anne Egitim Durumu',
    'Anne Sektor', 'Baba Egitim Durumu', 'Baba Sektor',
    'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
    'Spor Dalindaki Rolunuz Nedir?',
    'Aktif olarak bir STK üyesi misiniz?',
    'Ingilizce Biliyor musunuz?', 'Burs Aliyor mu?', 'Burs Aldigi Baska Kurum_categorical'
]

# Create a dictionary for ordinal encoding
ordinal_encoding = {
    '3.50 - 4.00': 5,
    '3.00 - 3.49': 4,
    '3.00 - 3.50': 4,
    '2.50 - 2.99': 3,
    '2.50 - 3.00': 3,
    '1.80 - 2.49': 2,
    '1.00 - 2.50': 2,
    '2.00 - 2.50': 2,
    'Hazırlığım': 1,
    'Not ortalaması yok': 1,
    '0 - 1.79': 0
}

# Apply ordinal encoding
df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].map(ordinal_encoding)

df['Kardes Sayisi'] = 5 - df['Kardes Sayisi']

# Create a new column with encoded values
df['Girisimcilikle Ilgili Deneyiminiz Var Mi?'] = df['Girisimcilikle Ilgili Deneyiminiz Var Mi?'].map({'Evet': 10, 'Hayır': 0})

# Create a new column with encoded values
df['Profesyonel Bir Spor Daliyla Mesgul musunuz?'] = df['Profesyonel Bir Spor Daliyla Mesgul musunuz?'].map({'Evet': 10, 'Hayır': 0})

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and target in the training set
X_train = train_df.drop('Degerlendirme Puani', axis=1)
y_train = train_df['Degerlendirme Puani']

# Initialize the TargetEncoder
encoder = TargetEncoder(cols=columns_to_encode)

# Fit the encoder on the training data and transform both train and test
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(test_df.drop('Degerlendirme Puani', axis=1))

print(X_train_encoded.info())

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(GradientBoostingRegressor(), X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert MSE to RMSE
rmse_scores = np.sqrt(-cv_scores)

print(f"Cross-validation RMSE scores: {rmse_scores}")
print(f"Mean RMSE: {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores) * 2:.4f})")

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_test = test_df['Degerlendirme Puani']
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Function to preprocess new data consistently
def preprocess_df(df):
    df_encoded = encoder.transform(df)
    df_scaled = scaler.transform(df_encoded)
    return df_scaled

# Function to make predictions on new data
def predict_df(model, df):
    df_preprocessed = preprocess_df(df)
    return model.predict(df_preprocessed)

### Save the model and preprocessing objects
import pickle

def save_model(model, encoder, scaler, feature_names, filename='model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((model, encoder, scaler, feature_names), file)
    print(f"Model saved as {filename}")

# Load the model and preprocessing objects
def load_model(filename='model.pkl'):
    with open(filename, 'rb') as file:
        model, encoder, scaler = pickle.load(file)
    return model, encoder, scaler

save_model(model, encoder, scaler, X_train_encoded.columns.to_list())


""" nn. implementation



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on validation set
val_mse = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Validation RMSE: {np.sqrt(val_mse):.4f}")

"""