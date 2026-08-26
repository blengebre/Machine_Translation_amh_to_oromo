from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import datetime

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_excel("cleaned_amh_omo.xlsx")

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)

# =====================================================
# SAVE TEXT FILES FOR SENTENCEPIECE
# =====================================================

with open("amharic.txt", "w", encoding="utf-8") as f:
    for s in train_df["Amharic"]:
        f.write(s + "\n")

with open("oromo.txt", "w", encoding="utf-8") as f:
    for s in train_df["Oromo"]:
        f.write(s + "\n")

# =====================================================
# TRAIN SENTENCEPIECE TOKENIZERS
# =====================================================

spm.SentencePieceTrainer.train(
    input="amharic.txt",
    model_prefix="am",
    vocab_size=4000,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

spm.SentencePieceTrainer.train(
    input="oromo.txt",
    model_prefix="om",
    vocab_size=4000,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

# =====================================================
# LOAD TOKENIZERS
# =====================================================

am_sp = spm.SentencePieceProcessor()
am_sp.load("am.model")

om_sp = spm.SentencePieceProcessor()
om_sp.load("om.model")

am_vocab_size = am_sp.get_piece_size()
om_vocab_size = om_sp.get_piece_size()

# =====================================================
# MODEL SETTINGS
# =====================================================

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": float(self.d_model),
            "warmup_steps": self.warmup_steps,
        }

learning_rate_schedule = CustomSchedule(d_model=128)

d_model = 128
num_heads = 4
ff_dim = 512

# =====================================================
# TOKENIZATION (REVERSED: OROMO -> AMHARIC)
# =====================================================

X_train = [
    om_sp.encode(s, out_type=int)
    for s in train_df["Oromo"]
]

y_train = [
    am_sp.encode(s, out_type=int)
    for s in train_df["Amharic"]
]

X_val = [
    om_sp.encode(s, out_type=int)
    for s in val_df["Oromo"]
]

y_val = [
    am_sp.encode(s, out_type=int)
    for s in val_df["Amharic"]
]

# =====================================================
# PADDING
# =====================================================

max_in = max(len(x) for x in X_train)
max_out = max(len(y) for y in y_train)

X_train = pad_sequences(X_train, maxlen=max_in, padding="post")
X_val   = pad_sequences(X_val, maxlen=max_in, padding="post")

y_train = pad_sequences(y_train, maxlen=max_out, padding="post")
y_val   = pad_sequences(y_val, maxlen=max_out, padding="post")

# Decoder inputs and targets
decoder_input_train = y_train[:, :-1]
decoder_target_train = y_train[:, 1:]

decoder_input_val = y_val[:, :-1]
decoder_target_val = y_val[:, 1:]

# =====================================================
# POSITIONAL ENCODING
# =====================================================

def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (i // 2)) / np.float32(d_model)
    )

    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(
        angle_rads[np.newaxis, ...],
        dtype=tf.float32
    )

# =====================================================
# TRANSFORMER BLOCK
# =====================================================

def transformer_block(x, num_heads, d_model, ff_dim, dropout=0.2):
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)

    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(attn))

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(d_model)(ffn)

    x = LayerNormalization(epsilon=1e-6)(x + Dropout(dropout)(ffn))

    return x

def encoder(x, num_layers=2):
    for _ in range(num_layers):
        x = transformer_block(x, num_heads, d_model, ff_dim)
    return x

def decoder_block(x, enc_output):
    # masked self attention
    attn1 = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x, use_causal_mask=True)
    x = LayerNormalization(epsilon=1e-6)(x + attn1)

    # cross attention
    attn2 = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, enc_output)

    x = LayerNormalization(epsilon=1e-6)(x + attn2)

    # feed forward
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(d_model)(ffn)

    x = LayerNormalization(epsilon=1e-6)(x + ffn)

    return x

# Encoder (Takes Oromo as input)
encoder_inputs = Input(shape=(max_in,))
enc_embed = Embedding(om_vocab_size, d_model)(encoder_inputs)
enc_embed += positional_encoding(max_in, d_model)

enc_output = encoder(enc_embed, num_layers=2)

# Decoder (Outputs Amharic)
decoder_inputs = Input(shape=(max_out-1,))
dec_embed = Embedding(am_vocab_size, d_model)(decoder_inputs)
dec_embed += positional_encoding(max_out-1, d_model)

dec_output = decoder_block(dec_embed, enc_output)

outputs = Dense(am_vocab_size, activation="softmax")(dec_output)

model = Model([encoder_inputs, decoder_inputs], outputs)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate_schedule, 
    beta_1=0.9, 
    beta_2=0.98, 
    epsilon=1e-9
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# TensorBoard callback
log_dir = "logs/fit/back_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,          # Wait 3 epochs before stopping
    restore_best_weights=True # Keep the best model, not the overfitted one
)

history = model.fit(
    [X_train, decoder_input_train],
    decoder_target_train,
    validation_data=(
        [X_val, decoder_input_val],
        decoder_target_val
    ),
    batch_size=16,
    epochs=20,
    callbacks=[tensorboard_callback, early_stopping]
)

print("TensorBoard log directory:", log_dir)

# =====================================================
# TRAIN
# =====================================================

model.save("back_translation_model.keras")
print("Model saved successfully!")

# =====================================================
# INFERENCE FUNCTION
# =====================================================

def translate(sentence):
    encoded = om_sp.encode(sentence, out_type=int)
    encoded = pad_sequences([encoded], maxlen=max_in, padding="post")
    
    output = [am_sp.bos_id()]
    
    for _ in range(max_out - 1):
        dec_input = pad_sequences([output], maxlen=max_out - 1, padding="post")
        preds = model.predict([encoded, dec_input], verbose=0)
        pos = len(output) - 1
        next_token = np.argmax(preds[0, pos, :])
        
        if next_token == am_sp.eos_id():
            break
        
        output.append(int(next_token))
    
    tokens = [t for t in output if t not in (am_sp.bos_id(), am_sp.eos_id(), 0)]
    return am_sp.decode(tokens)


# =====================================================
# BLEU EVALUATION
# =====================================================

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smooth = SmoothingFunction().method4

def compute_bleu(df, n=100):
    sample_size = min(n, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    scores = []
    for idx, row in sample_df.iterrows():
        try:
            ref = row["Amharic"].split()
            pred = translate(row["Oromo"]).split()
            
            if ref and pred:
                scores.append(sentence_bleu([ref], pred, smoothing_function=smooth))
        except Exception as e:
            print(f"Skipping row {idx}: {e}")
            continue

    return float(np.mean(scores)) if scores else 0.0


print("\nComputing BLEU score on test set...")
bleu_score = compute_bleu(test_df, n=min(100, len(test_df)))
print(f"BLEU score: {bleu_score:.4f}")
