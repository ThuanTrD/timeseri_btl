import numpy as np
import h5py
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
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
csv_path = r"D:\Du_doan_bao\train_data_with_pixels.csv"
h5_path = r"D:\Du_doan_bao\pixel_sequences.h5"
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
        y.append(labels[i + seq_length])  # Nhãn tại thời điểm cuối
        indices.append(i)
    return np.array(X_meta), np.array(y), np.array(indices)

seq_length = 6
X_meta_seq, y_seq, indices_seq = create_sequences(meta_features_scaled, labels, seq_length)

# Chia tập train/test
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
            batch_meta = meta_data[batch_indices]
            batch_labels = labels[batch_indices]  # Sử dụng nhãn gốc
            yield (batch_pixel, batch_meta), np.array(batch_labels, dtype=np.int8)

# Tạo tf.data.Dataset
def create_dataset(h5_path, meta_data, labels, indices, seq_length, batch_size, is_training=True):
    output_signature = (
        tf.TensorSpec(shape=(None, seq_length, 64, 64, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, seq_length, meta_data.shape[2]), dtype=tf.float32),
    ), tf.TensorSpec(shape=(None,), dtype=tf.int8)
    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(h5_path, meta_data, labels, indices, seq_length, batch_size),
        output_signature=output_signature
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Xây dựng mô hình Temporal Fusion Transformer
def build_tft_model(seq_length, pixel_shape=(64, 64, 1), meta_features=5):
    # Input layers
    pixel_input = Input(shape=(seq_length, *pixel_shape), name='pixel_input')
    meta_input = Input(shape=(seq_length, meta_features), name='meta_input')

    # CNN để trích xuất đặc trưng từ pixel
    x = tf.keras.layers.TimeDistributed(
        tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten()
        ])
    )(pixel_input)
    x = tf.keras.layers.TimeDistributed(Dense(32, activation='relu'))(x)

    # Temporal Fusion Transformer block
    # Multi-Head Attention
    attention_output = MultiHeadAttention(key_dim=32, num_heads=4)(x, x)
    x = LayerNormalization(epsilon=1e-6)(attention_output + x)  # Add & Norm

    # Gated Linear Unit (GLU) simulation
    x = Dense(64, activation='relu')(x)
    gate = Dense(64, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, gate])  # Gated activation

    # Kết hợp với meta data
    meta_processed = Dense(32, activation='relu')(meta_input)
    combined = Concatenate()([x, meta_processed])

    # Giảm dimension để dự đoán nhãn đơn
    z = GlobalAveragePooling1D()(combined)  # Giảm từ (batch, seq_length, features) xuống (batch, features)
    z = Dense(16, activation='relu')(z)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[pixel_input, meta_input], outputs=output)
    return model

# Tạo và biên dịch mô hình
model = build_tft_model(seq_length, pixel_shape=(64, 64, 1), meta_features=meta_features.shape[1])
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
model.save('tft_landfall_model.h5')