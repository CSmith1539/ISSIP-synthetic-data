import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda

# Vae subclass
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, binary_mask, kl_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.binary_mask = tf.constant(binary_mask, dtype=tf.bool)

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_recon, z_mean, z_log_var = self(x, training=True)

            # Split binary vs numeric components
            x_bin = tf.boolean_mask(x, self.binary_mask, axis=1)
            x_recon_bin = tf.boolean_mask(x_recon, self.binary_mask, axis=1)

            x_num = tf.boolean_mask(x, ~self.binary_mask, axis=1)
            x_recon_num = tf.boolean_mask(x_recon, ~self.binary_mask, axis=1)

            # Compute binary and numeric reconstruction losses
            recon_bin = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x_bin, x_recon_bin)
            )
            recon_num = tf.reduce_mean(tf.square(x_num - x_recon_num))

            recon_loss = recon_bin + recon_num

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )

            total_loss = recon_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "recon_bin": recon_bin,
            "recon_num": recon_num,
            "kl_loss": kl_loss,
        }

# Infer decimal precision per column
def infer_decimals(series):
    if not np.issubdtype(series.dtype, np.number):
        return None

    s = series.dropna().astype(str)

    # Extract decimal parts
    decimals = s.str.extract(r'\.(\d*)')[0].dropna()
    if len(decimals) == 0:
        return 0  # integer column

    # Get the max number of digits after decimal point
    return decimals.map(len).max()

def generate_data(df, one_hot, n_samples, LATENT_DIM=8, EPOCHS=50, BATCH_SIZE=32, kl_weight=0.001):
    print(f'Running with latent_dim={LATENT_DIM}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, kl_weight={kl_weight}')

    rounding_map = {col: infer_decimals(df[col]) for col in df.columns}
    
    # Detect binary columns
    binary_mask = np.array([(df[col].dropna().isin([0, 1]).all()) for col in df.columns])

    # Normalize all columns to [0, 1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    #data_scaled = scaler.fit_transform(df[~binary_mask])

    input_dim = data_scaled.shape[1]

    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(128, activation='relu')(inputs)
    h = layers.Dense(64, activation='relu')(h)
    h = layers.BatchNormalization()(h)
    z_mean = layers.Dense(LATENT_DIM)(h)
    z_log_var = layers.Dense(LATENT_DIM)(h)

    # This can be adapted for different datasets
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(64, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_dim, activation='sigmoid')(x) # Replaced linear
    decoder = models.Model(latent_inputs, outputs, name="decoder")

    # Set up VAE
    vae = VAE(encoder, decoder, binary_mask, kl_weight)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # Train
    vae.fit(data_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # Generate data
    n_samples = 1000  # number of synthetic samples to generate
    z_new = np.random.normal(size=(n_samples, LATENT_DIM))
    generated_scaled = decoder.predict(z_new)
    generated = scaler.inverse_transform(generated_scaled)

    # Convert back to DataFrame
    generated_df = pd.DataFrame(generated, columns=df.columns)

    # Enforce one-hot integrity
    for group in one_hot:
        subset = generated_df[group]
        idx_max = subset.values.argmax(axis=1)  # index of max per row
        subset[:] = 0
        subset.values[np.arange(len(subset)), idx_max] = 1
        generated_df[group] = subset

    # Round columns to same significance as before
    for col, n_decimals in rounding_map.items():
        if n_decimals is not None and n_decimals > 0:
            generated_df[col] = generated_df[col].round(n_decimals)
        elif n_decimals == 0:
            generated_df[col] = generated_df[col].round().astype(int)

    return generated_df

if __name__ == "__main__":
    generate_data(LATENT_DIM=8, EPOCHS=30, BATCH_SIZE=24, kl_weight=0.01)

    print("Script completed")


