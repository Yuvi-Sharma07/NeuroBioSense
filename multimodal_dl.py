import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ================= PARAMETERS =================
TRAIN_MODEL = True
VIDEO_FRAMES = 16
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3
BIOSIGNAL_LENGTH = 1280
BIOSIGNAL_CHANNELS = 6
NUM_CLASSES = 7
BATCH_SIZE = 8

BASE_DIR = r"D:\Deep Learning\NeuroBioSense"
VIDEO_DIR = os.path.join(BASE_DIR, "Advertisement Categories")
BIOSIGNAL_RAW_DIR = os.path.join(BASE_DIR, "Biosignal Files", "Raw")

EMOTION_MAP = {'A': 0, 'D': 1, 'F': 2, 'H': 3, 'N': 4, 'SA': 5, 'SU': 6}

REVERSE_EMOTION_MAP = {
    0: 'Anger', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

# ================= NEW: RESAMPLING FUNCTION =================
def resample_signal(signal, target_length):
    original_length = signal.shape[0]

    if original_length == target_length:
        return signal

    new_indices = np.linspace(0, original_length - 1, target_length)
    resampled = np.zeros((target_length, signal.shape[1]))

    for i in range(signal.shape[1]):
        resampled[:, i] = np.interp(new_indices, np.arange(original_length), signal[:, i])

    return resampled


# ================= VIDEO =================
def extract_video_frames(video_path, num_frames=VIDEO_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))

    step = max(1, total_frames // num_frames)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, CHANNELS), dtype=np.uint8))

    cap.release()
    return np.array(frames) / 255.0


# ================= GLOBAL CACHE FOR BIOSIGNALS =================
_BIOSIGNAL_CACHE = None

def get_biosignal_cache():
    global _BIOSIGNAL_CACHE
    if _BIOSIGNAL_CACHE is None:
        csv_path = os.path.join(BASE_DIR, "Biosignal Files", "Pre-Processed", "32-Hertz.csv")
        if os.path.exists(csv_path):
            print(f"\nLoading biosignal cache from {csv_path} (this might take a moment)...")
            df = pd.read_csv(csv_path)
            _BIOSIGNAL_CACHE = {emo: df[df['EMOTION'] == emo].copy() for emo in df['EMOTION'].unique()}
        else:
            print(f"[ERROR] Could not find {csv_path}")
            _BIOSIGNAL_CACHE = {}
    return _BIOSIGNAL_CACHE

def load_biosignals(emotion):
    cache = get_biosignal_cache()

    # Map 'H' (Happy) from folder to 'J' (Joy) in CSV
    csv_emotion = 'J' if emotion == 'H' else emotion

    if csv_emotion not in cache or len(cache[csv_emotion]) < BIOSIGNAL_LENGTH:
        print(f"[ERROR] No biosignal data for emotion {csv_emotion}")
        return np.zeros((BIOSIGNAL_LENGTH, BIOSIGNAL_CHANNELS), dtype=np.float32)

    df_emo = cache[csv_emotion]

    # Randomly pick a starting index to extract a chunk
    max_start = len(df_emo) - BIOSIGNAL_LENGTH
    start_idx = np.random.randint(0, max_start + 1)

    chunk = df_emo.iloc[start_idx:start_idx + BIOSIGNAL_LENGTH]

    # Extract columns in expected order: ACC (X,Y,Z), BVP, EDA, TEMP
    biosignal = chunk[['X', 'Y', 'Z', 'BVP', 'EDA', 'TEMP']].values

    # Fix NaN
    biosignal = np.nan_to_num(biosignal)

    # Normalize
    min_val = np.min(biosignal)
    max_val = np.max(biosignal)
    if max_val - min_val != 0:
        biosignal = (biosignal - min_val) / (max_val - min_val)

    return biosignal.astype(np.float32)


# ================= DATA COLLECTION =================
def get_all_samples():
    samples = []
    for category in os.listdir(VIDEO_DIR):
        cat_path = os.path.join(VIDEO_DIR, category)
        if not os.path.isdir(cat_path): continue

        for participant in os.listdir(cat_path):
            part_path = os.path.join(cat_path, participant)
            if not os.path.isdir(part_path): continue

            for ad in os.listdir(part_path):
                ad_path = os.path.join(part_path, ad)
                if not os.path.isdir(ad_path): continue

                for emotion in os.listdir(ad_path):
                    emo_path = os.path.join(ad_path, emotion)
                    if not os.path.isdir(emo_path): continue

                    label = EMOTION_MAP.get(emotion.upper(), -1)
                    if label == -1: continue

                    for video_file in os.listdir(emo_path):
                        if video_file.endswith(".mp4"):
                            video_path = os.path.join(emo_path, video_file)
                            samples.append({
                                'video_path': video_path,
                                'emotion': emotion.upper(),
                                'label': label
                            })
    return samples

# ================= GENERATOR =================
def create_generator(samples):
    def generator():
        for sample in samples:
            frames = extract_video_frames(sample['video_path'])
            biosignals = load_biosignals(sample['emotion'])
            yield (frames, biosignals), sample['label']
    return generator

# ================= DATA AUGMENTATION =================
def augment_video(inputs, label):
    video_frames, biosignals = inputs

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        video_frames = tf.reverse(video_frames, axis=[2])

    # Random brightness
    video_frames = video_frames + tf.random.uniform((), -0.1, 0.1)
    video_frames = tf.clip_by_value(video_frames, 0.0, 1.0)

    # Random contrast
    factor = tf.random.uniform((), 0.8, 1.2)
    mean = tf.reduce_mean(video_frames, axis=[1, 2, 3], keepdims=True)
    video_frames = (video_frames - mean) * factor + mean
    video_frames = tf.clip_by_value(video_frames, 0.0, 1.0)

    # Add slight noise to biosignals
    noise = tf.random.normal(tf.shape(biosignals), mean=0.0, stddev=0.02)
    biosignals = biosignals + noise

    return (video_frames, biosignals), label

# ================= DATASET =================
def get_dataset(samples, augment=False):
    dataset = tf.data.Dataset.from_generator(
        create_generator(samples),
        output_signature=(
            (
                tf.TensorSpec(shape=(VIDEO_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS), dtype=tf.float32),
                tf.TensorSpec(shape=(BIOSIGNAL_LENGTH, BIOSIGNAL_CHANNELS), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    # Apply label smoothing: instead of hard [0,0,1,0,...] use [0.01,0.01,0.93,0.01,...]
    def smooth_labels(x, y):
        y_smooth = tf.one_hot(y, NUM_CLASSES)
        y_smooth = y_smooth * 0.9 + 0.1 / NUM_CLASSES
        return x, y_smooth

    dataset = dataset.map(smooth_labels)

    if augment:
        dataset = dataset.map(augment_video, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(256)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ================= MODEL =================
def build_multimodal_model():
    video_input = tf.keras.Input(shape=(VIDEO_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))
    base_cnn = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False,
                                                 input_shape=(FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))

    # Freeze most of backbone — only unfreeze last 10 layers
    for layer in base_cnn.layers[:-10]:
        layer.trainable = False

    x = tf.keras.layers.TimeDistributed(base_cnn)(video_input)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    video_features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)

    bio_input = tf.keras.Input(shape=(BIOSIGNAL_LENGTH, BIOSIGNAL_CHANNELS))
    y = tf.keras.layers.Conv1D(32, 3, activation='relu')(bio_input)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling1D(2)(y)
    y = tf.keras.layers.Conv1D(64, 3, activation='relu')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling1D(2)(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(y)
    bio_features = tf.keras.layers.Dense(64, activation='relu')(y)

    fused = tf.keras.layers.Concatenate()([video_features, bio_features])
    fused = tf.keras.layers.Dense(256, activation='relu')(fused)
    fused = tf.keras.layers.BatchNormalization()(fused)
    fused = tf.keras.layers.Dropout(0.5)(fused)
    fused = tf.keras.layers.Dense(128, activation='relu')(fused)
    fused = tf.keras.layers.Dropout(0.4)(fused)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(fused)

    model = tf.keras.Model(inputs=[video_input, bio_input], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ================= COMPUTE CLASS WEIGHTS =================
def compute_class_weights(samples):
    labels = [s['label'] for s in samples]
    count = Counter(labels)
    total = len(labels)
    n_classes = len(count)
    weights = {}
    for cls, cnt in count.items():
        weights[cls] = total / (n_classes * cnt)
    return weights


if __name__ == "__main__":
    print("Initializing model...")
    model = build_multimodal_model()

    print("Gathering dataset samples...")
    all_samples = get_all_samples()
    print(f"Total samples found: {len(all_samples)}")

    labels = [s['label'] for s in all_samples]
    count = Counter(labels)
    print("\nClass Distribution:")
    for k, v in count.items():
        print(f"  {REVERSE_EMOTION_MAP[k]}: {v}")

    # 70% Train, 15% Val, 15% Test (stratified by label)
    all_labels = [s['label'] for s in all_samples]
    train_samples, temp_samples = train_test_split(all_samples, test_size=0.3, random_state=42, stratify=all_labels)
    temp_labels = [s['label'] for s in temp_samples]
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42, stratify=temp_labels)

    print(f"Train: {len(train_samples)}, Validation: {len(val_samples)}, Test: {len(test_samples)}")

    train_dataset = get_dataset(train_samples, augment=True)
    val_dataset = get_dataset(val_samples, augment=False)
    test_dataset = get_dataset(test_samples, augment=False)

    class_weights = compute_class_weights(train_samples)
    print(f"Class Weights: {class_weights}")

    weights_path = os.path.join(BASE_DIR, "multimodal_weights.weights.h5")

    # ================= LOAD OR TRAIN =================
    if os.path.exists(weights_path):
        print("\n[OK] Loading pre-trained weights...")
        model.load_weights(weights_path)
        print("Model loaded successfully. No training needed.")
    else:
        print("\n[!] No saved model found. Training now...")
        train_steps = max(1, len(train_samples) // BATCH_SIZE)
        val_steps = max(1, len(val_samples) // BATCH_SIZE)

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
            ),
        ]

        model.fit(
            train_dataset.repeat(),
            validation_data=val_dataset.repeat(),
            epochs=40,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            class_weight=class_weights
        )
        model.save_weights(weights_path)
        print("Model trained and saved successfully!")

    print("\n[INFO] Evaluating model on Test Set...")
    # ================= EVALUATION =================
    test_steps = max(1, len(test_samples) // BATCH_SIZE)
    test_loss, test_acc = model.evaluate(test_dataset, steps=test_steps)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

    # ================= COLLECT PREDICTIONS FOR ROC =================
    print("\n[INFO] Collecting predictions for ROC curve...")
    all_true_labels = []
    all_pred_probs = []

    for batch_inputs, batch_labels in test_dataset:
        preds = model.predict(batch_inputs, verbose=0)
        all_pred_probs.append(preds)
        all_true_labels.append(batch_labels.numpy())

    all_true_labels = np.concatenate(all_true_labels, axis=0)   # (N, NUM_CLASSES) one-hot smoothed
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)     # (N, NUM_CLASSES) softmax

    # Convert smoothed one-hot back to hard labels for ROC
    true_binary = (all_true_labels > 0.5).astype(int)

    # ================= PLOT ROC CURVES =================
    print("[INFO] Plotting ROC curves...")

    # --- Color palette for 7 emotions ---
    colors = ['#e74c3c', '#8e44ad', '#2980b9', '#27ae60', '#7f8c8d', '#e67e22', '#f1c40f']

    plt.figure(figsize=(10, 8))

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    for i in range(NUM_CLASSES):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(true_binary[:, i], all_pred_probs[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

        plt.plot(
            fpr_dict[i], tpr_dict[i],
            color=colors[i], linewidth=2,
            label=f"{REVERSE_EMOTION_MAP[i]} (AUC = {roc_auc_dict[i]:.3f})"
        )

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)

    plt.plot(
        all_fpr, mean_tpr,
        color='navy', linewidth=3, linestyle='--',
        label=f"Macro-Average (AUC = {macro_auc:.3f})"
    )

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle=':')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve — Multimodal Emotion Recognition (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_save_path = os.path.join(BASE_DIR, "roc_curve.png")
    plt.savefig(roc_save_path, dpi=150)
    plt.show()
    print(f"\n[OK] ROC curve saved to: {roc_save_path}")

    # Print AUC summary
    print("\n========== AUC Summary ==========")
    for i in range(NUM_CLASSES):
        print(f"  {REVERSE_EMOTION_MAP[i]:>10}: {roc_auc_dict[i]:.4f}")
    print(f"  {'Macro-Avg':>10}: {macro_auc:.4f}")
    print("=================================")

    print("\n[INFO] Running prediction demo on a single test batch...")
    # ================= PREDICTION =================
    for batch_inputs, batch_labels in test_dataset.take(1):
        predictions = model.predict(batch_inputs)

        for i in range(len(predictions)):
            probs = predictions[i]
            pred_idx = int(np.argmax(probs))
            true_idx = int(np.argmax(batch_labels[i].numpy()))

            print("\n----------------------------")
            print(f"Sample {i+1}")
            print("Probabilities:", np.round(probs, 2))
            print("Predicted Emotion:", REVERSE_EMOTION_MAP[pred_idx])
            print("Actual Emotion:", REVERSE_EMOTION_MAP[true_idx])
            print("----------------------------")

        break

