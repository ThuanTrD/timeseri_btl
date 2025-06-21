import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Tắt cảnh báo loky
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Định nghĩa cột và kiểu dữ liệu
meta_cols = ['year', 'month', 'day', 'hour', 'grade', 'lat', 'lng', 'pressure', 'wind', 'landfall']
pixel_cols = [f'pixel_{i}' for i in range(4096)]
dtype_dict = {col: 'float32' for col in meta_cols if col != 'landfall'}
dtype_dict.update({'landfall': 'int8'})
dtype_dict.update({col: 'uint8' for col in pixel_cols})

# Đọc file CSV theo chunk và lưu pixel vào HDF5
csv_path = r"D:\time_seri\Du_doan_bao\train_data_with_pixels.csv"
h5_path = r"D:\time_seri\Du_doan_bao\pixel_sequences.h5"

chunk_size = 1000

if not os.path.exists(h5_path):
    with h5py.File(h5_path, 'w') as f:
        dset = None
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=dtype_dict):
            chunk = chunk[(chunk['pressure'] > 0) & (chunk['wind'] > 0)]
            if len(chunk) == 0:
                continue
            pixel_data = chunk[pixel_cols].values.reshape(-1, 64, 64, 1).astype(np.float32) / 255.0
            meta_data = chunk[meta_cols].values
            if dset is None:
                dset = f.create_dataset('pixels', shape=(0, 64, 64, 1), maxshape=(None, 64, 64, 1), dtype=np.float32)
                f.create_dataset('meta', shape=(0, len(meta_cols)), maxshape=(None, len(meta_cols)), dtype=np.float32)
                f.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int8)
            curr_size = dset.shape[0]
            dset.resize(curr_size + pixel_data.shape[0], axis=0)
            f['meta'].resize(curr_size + meta_data.shape[0], axis=0)
            f['labels'].resize(curr_size + meta_data.shape[0], axis=0)
            dset[curr_size:] = pixel_data
            f['meta'][curr_size:] = meta_data
            f['labels'][curr_size:] = chunk['landfall'].values

# Tải metadata và nhãn từ HDF5
with h5py.File(h5_path, 'r') as f:
    meta_features = f['meta'][:][:, 5:]  # Lấy lat, lng, pressure, wind, grade
    labels = f['labels'][:]

# Chuẩn hóa metadata
scaler = MinMaxScaler()
meta_features_scaled = scaler.fit_transform(meta_features)
joblib.dump(scaler, 'scaler.pkl')

# Tạo chuỗi thời gian và lưu chỉ số
def create_sequences(meta_data, labels, seq_length):
    X_meta, y, indices = [], [], []
    for i in range(len(meta_data) - seq_length):
        X_meta.append(meta_data[i:i+seq_length])
        y.append(labels[i + seq_length])
        indices.append(i)
    return np.array(X_meta), np.array(y), np.array(indices)

seq_length = 6
X_meta_seq, y_seq, indices_seq = create_sequences(meta_features_scaled, labels, seq_length)

# Chia tập train/test (sử dụng dữ liệu gốc)
train_indices, test_indices = train_test_split(
    np.arange(len(indices_seq)), test_size=0.2, random_state=42, stratify=y_seq
)
X_meta_train = X_meta_seq[train_indices]
X_meta_test = X_meta_seq[test_indices]
y_train = y_seq[train_indices]
y_test = y_seq[test_indices]
train_indices_seq = indices_seq[train_indices]
test_indices_seq = indices_seq[test_indices]

# Generator cho chuỗi thời gian
def sequence_generator(h5_path, meta_data, labels, indices, seq_length, batch_size):
    with h5py.File(h5_path, 'r') as f:
        pixel_data = f['pixels']
        while True:
            batch_indices = np.random.permutation(len(indices))[:batch_size]
            batch_pixel = np.array([pixel_data[i:i+seq_length] for i in indices[batch_indices]], dtype=np.float32)
            batch_meta = meta_data[batch_indices]  # Sử dụng trực tiếp mảng đã là chuỗi
            batch_labels = labels[batch_indices]  # Lấy nhãn trực tiếp từ batch_indices
            yield (batch_pixel, batch_meta), np.array(batch_labels, dtype=np.int8)

# Tạo tf.data.Dataset
def create_dataset(h5_path, meta_data, labels, indices, seq_length, batch_size, is_training=True):
    output_signature = (
        tf.TensorSpec(shape=(None, seq_length, 64, 64, 1), dtype=tf.float32),  # Pixel data
        tf.TensorSpec(shape=(None, seq_length, meta_data.shape[2]), dtype=tf.float32),  # Meta data
    ), tf.TensorSpec(shape=(None,), dtype=tf.int8)  # Labels
    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(h5_path, meta_data, labels, indices, seq_length, batch_size),
        output_signature=output_signature
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Xây dựng mô hình CNN-LSTM
pixel_input = Input(shape=(seq_length, 64, 64, 1), name='pixel_input')
x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(pixel_input)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Flatten())(x)
x = TimeDistributed(Dense(32))(x)
x = LSTM(16, activation='tanh')(x)

meta_input = Input(shape=(seq_length, meta_features.shape[1]), name='meta_input')
y = LSTM(16, activation='tanh', return_sequences=True)(meta_input)
y = LSTM(8, activation='tanh')(y)

combined = Concatenate()([x, y])
z = Dropout(0.3)(combined)
z = Dense(8, activation='relu')(z)
output = Dense(1, activation='sigmoid')(z)  # Giữ sigmoid, áp ngưỡng sau

model = Model(inputs=[pixel_input, meta_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Tạo dataset
batch_size = 64
train_dataset = create_dataset(h5_path, X_meta_train, y_train, train_indices_seq, seq_length, batch_size, is_training=True)
val_dataset = create_dataset(h5_path, X_meta_test, y_test, test_indices_seq, seq_length, batch_size, is_training=False)

# Huấn luyện
train_steps = len(train_indices) // batch_size
val_steps = len(test_indices) // batch_size
history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,
    epochs=30,
    validation_data=val_dataset,
    validation_steps=val_steps,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)

# Lưu mô hình
model.save('lstm_cnn_landfall_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Thêm phần dự đoán để kiểm tra
with h5py.File(h5_path, 'r') as f:
    pixel_data = f['pixels'][:]

# Lấy một batch dữ liệu test để dự đoán
test_pixel_batch = np.array([pixel_data[i:i+seq_length] for i in test_indices_seq[:batch_size]])
test_meta_batch = X_meta_test[:batch_size]

# Chuẩn hóa meta data
test_meta_batch_scaled = scaler.transform(test_meta_batch.reshape(-1, 5)).reshape(batch_size, seq_length, 5)

# Dự đoán và áp ngưỡng để lấy nhãn
predictions = model.predict([test_pixel_batch, test_meta_batch_scaled])
predicted_labels = (predictions > 0.5).astype(int)  # Áp ngưỡng 0.5 để phân loại
print("Predicted landfall labels for test batch:", predicted_labels.flatten())