# task8_regularization_overfitting_final.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import psutil

# configuration
DROPOUT_RATE = 0.4
EPOCHS = 30
BATCH_SIZE = 32
PATIENCE = 5

# load processed data
print("Task : Regularization & Overfitting Analysis")
print("\n Loading processed data from .npy files...")

X = np.load('X_token_matrix.npy')  # Shape: (n_samples, 512)
y_raw = np.load('y_labels.npy')  

print(f"X shape: {X.shape}")
print(f"y shape: {y_raw.shape}")

# Reshape for transformer [batch_size, seq_len, feature_dim]
X = np.expand_dims(X, axis=-1)  # Shape: (n_samples, 512, 1)
print(f"Reshaped for Transformer: {X.shape}")

# encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)
print(f" Classes: {dict(zip(range(num_classes), le.classes_))}")

# imbalance handling
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# split data
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

# transformer model with dropout
def build_transformer(input_shape, num_classes, dropout_rate=0.4):
    """Transformer Encoder with Dropout regularization"""
    inputs = Input(shape=input_shape)
    
    # Input projection
    x = Dense(128, activation='relu')(inputs)
    
    # Multi-Head Attention with Dropout
    attention_output = MultiHeadAttention(
        num_heads=8,
        key_dim=128,
        dropout=dropout_rate  # Dropout on attention weights
    )(x, x, training=True)
    
    # Skip connection + LayerNorm
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    
    # Network with Dropout
    ffn = tf.keras.Sequential([
        Dense(256, activation='relu'),
        Dropout(dropout_rate),  # Dropout after dense
        Dense(128, activation='relu'),
        Dropout(dropout_rate)   # Dropout before residual
    ])
    ffn_output = ffn(x)
    
    # Skip connection + LayerNorm
    x = Add()([x, ffn_output])
    x = LayerNormalization()(x)
    
    # Global pooling + Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)  # Final Dropout
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

model = build_transformer(
    input_shape=X.shape[1:],
    num_classes=num_classes,
    dropout_rate=DROPOUT_RATE
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTransformer Model (with Dropout):")
model.summary()

# early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)
print(f"\n✅ EarlyStopping configured: Patience={PATIENCE} epochs")

# regularized training

print("\n Training with Dropout & EarlyStopping...")


# start monitoring time and memory
start_time = time.time()
process = psutil.Process()
memory_start = process.memory_info().rss / (1024 ** 2)  # in MB

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# End monitoring
end_time = time.time()
memory_end = process.memory_info().rss / (1024 ** 2)
training_time = end_time - start_time
memory_used = memory_end - memory_start

print(f"\n Training Time: {training_time:.2f} seconds")
print(f"Memory Used: {memory_used:.2f} MB")


# history saving and building the plot
# Save training history
history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv('task8_training_history.csv', index=False)
print(" saved: task8_training_history.csv")

# plot Loss & Accuracy curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['loss'], 'b--', label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
ax1.set_title('Loss Evolution - Overfitting Detection', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history['accuracy'], 'b--', label='Train Accuracy', linewidth=2)
ax2.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
ax2.set_title('Accuracy Evolution - Overfitting Detection', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task8_loss_accuracy_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("saved: task8_loss_accuracy_curves.png")

# overfitting report
print("overfitting analysis report")


final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
loss_gap = abs(final_train_loss - final_val_loss)

print(f"Final Train Loss: {final_train_loss:.4f}")
print(f"Final Val Loss:   {final_val_loss:.4f}")
print(f"Loss Gap:         {loss_gap:.4f}")

if loss_gap > 0.15:
    print("high overfitting")
elif loss_gap > 0.05:
    print("moderate overfiting")
else:
    print("optimal fit")

# confusion matrix 
print("\n confusion matrix")

y_pred = np.argmax(model.predict(X_val), axis=1)
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot and save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Validation Set')
plt.savefig('task8_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("saved: task8_confusion_matrix.png")


