import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda

# Config
CSV_FILE = "data/processed_data.csv"   # path to your CSV
SKIP_COLUMNS = []  # columns to exclude entirely
LATENT_DIM = 8                # latent space dimension
EPOCHS = 50                   # adjust for training time
BATCH_SIZE = 32
one_hot_groups = [
    ["Mjob_at_home","Mjob_health","Mjob_other","Mjob_services","Mjob_teacher"],
    ["Fjob_at_home","Fjob_health","Fjob_other","Fjob_services","Fjob_teacher"],
    ["reason_course","reason_home","reason_other","reason_reputation"],
    ["guardian_father", "guardian_mother", "guardian_other"],
]

# Vae subclass
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, kl_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var

    def train_step(self, data):
        x = data  # just use the input
        with tf.GradientTape() as tape:
            x_recon, z_mean, z_log_var = self(x, training=True)
            recon_loss = tf.reduce_mean(tf.square(x - x_recon))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

# Load Data
df = pd.read_csv(CSV_FILE)

# Drop unwanted columns
df = df.drop(columns=[c for c in SKIP_COLUMNS if c in df.columns])

# Drop non-numeric columns that remain
df = df.select_dtypes(include=[np.number])

# Fill missing values (mean imputation)
df = df.fillna(df.mean())

# Normalize all columns to [0, 1]
scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(df.values)
input_dim = data_scaled.shape[1]

print(f"Data loaded: {df.shape[0]} rows, {input_dim} features")

# Encoder
inputs = layers.Input(shape=(input_dim,))
h = layers.Dense(64, activation='relu')(inputs)
z_mean = layers.Dense(LATENT_DIM)(h)
z_log_var = layers.Dense(LATENT_DIM)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = layers.Input(shape=(LATENT_DIM,))
x = layers.Dense(64, activation='relu')(latent_inputs)
outputs = layers.Dense(input_dim, activation='sigmoid')(x)
decoder = models.Model(latent_inputs, outputs, name="decoder")

# Set up VAE
vae = VAE(encoder, decoder, kl_weight=0.001)
vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

# Train
vae.fit(data_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Generate data
n_samples = 100  # number of synthetic samples to generate
z_new = np.random.normal(size=(n_samples, LATENT_DIM))
generated_scaled = decoder.predict(z_new)
generated = scaler.inverse_transform(generated_scaled)

# Convert back to DataFrame
generated_df = pd.DataFrame(generated, columns=df.columns)

# Enforce one-hot integrity
for group in one_hot_groups:
    subset = generated_df[group]
    idx_max = subset.values.argmax(axis=1)  # index of max per row
    subset[:] = 0
    subset.values[np.arange(len(subset)), idx_max] = 1
    generated_df[group] = subset

# Round all other values (continuous features)
generated_df = generated_df.round().astype(int)

print(generated_df.head())

# Save to CSV
generated_df.to_csv("output/synthetic_output.csv", index=False)

print("Script completed")
