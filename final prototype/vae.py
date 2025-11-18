import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import binary_crossentropy, mse


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, binary_mask, kl_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.binary_mask = tf.constant(binary_mask, dtype=tf.bool)
        self.kl_weight = kl_weight

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_recon, z_mean, z_log_var = self(x, training=True)

            # Split into binary vs numeric (mask is order-aligned)
            x_bin = tf.boolean_mask(x, self.binary_mask, axis=1)
            x_recon_bin = tf.boolean_mask(x_recon, self.binary_mask, axis=1)
            x_num = tf.boolean_mask(x, ~self.binary_mask, axis=1)
            x_recon_num = tf.boolean_mask(x_recon, ~self.binary_mask, axis=1)

            # Compute individual losses safely in graph mode
            def compute_bin_loss():
                return tf.reduce_mean(binary_crossentropy(x_bin, x_recon_bin))
            def compute_num_loss():
                return tf.reduce_mean(tf.square(x_num - x_recon_num))

            recon_bin = tf.cond(tf.size(x_bin) > 0, compute_bin_loss, lambda: 0.0)
            recon_num = tf.cond(tf.size(x_num) > 0, compute_num_loss, lambda: 0.0)

            # Balance the losses by feature count
            n_bin = tf.reduce_sum(tf.cast(self.binary_mask, tf.float32))
            n_num = tf.reduce_sum(tf.cast(~self.binary_mask, tf.float32))
            denom = n_bin + n_num
            recon_loss = tf.where(denom > 0, (recon_bin * n_bin + recon_num * n_num) / denom, 0.0)

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

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

def infer_decimals(series):
    if not np.issubdtype(series.dtype, np.number):
        return None
    s = series.dropna().astype(str)
    decimals = s.str.extract(r'\.(\d*)')[0].dropna()
    if len(decimals) == 0:
        return 0
    return min(decimals.map(len).max(), 4)


def generate_data(df, one_hot, n_samples, LATENT_DIM=8, EPOCHS=50, BATCH_SIZE=32, kl_weight=1e-4):
    print(f'Running with latent_dim={LATENT_DIM}, epochs={EPOCHS}, kl_weight={kl_weight}')

    # Preserve original column order
    cols = list(df.columns)

    # Identify binary columns by content
    binary_mask = np.array([(df[c].dropna().isin([0, 1]).all()) for c in cols])
    numeric_mask = ~binary_mask

    # Scale numeric columns only
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    if np.any(numeric_mask):
        num_cols = [c for c, m in zip(cols, numeric_mask) if m]
        df_scaled.loc[:, num_cols] = scaler.fit_transform(df.loc[:, num_cols].astype(float))

    # Get numpy array in the original order
    data_scaled = df_scaled[cols].values.astype(np.float32)
    input_dim = data_scaled.shape[1]

    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(128, activation='relu')(inputs)
    h = layers.Dense(64, activation='relu')(h)
    z_mean = layers.Dense(LATENT_DIM)(h)
    z_log_var = layers.Dense(LATENT_DIM)(h)

    def sampling(args):
        z_mean, z_log_var = args
        eps = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM))
        return z_mean + K.exp(0.5 * z_log_var) * eps

    z = layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = layers.Input(shape=(LATENT_DIM,))
    d = layers.Dense(64, activation='relu')(latent_inputs)
    d = layers.Dense(128, activation='relu')(d)

    n_bin = int(np.sum(binary_mask))
    n_num = int(np.sum(numeric_mask))

    binary_outputs = layers.Dense(n_bin, activation='sigmoid')(d) if n_bin > 0 else None
    numeric_outputs = layers.Dense(n_num, activation='linear')(d) if n_num > 0 else None

    outputs_concat = binary_outputs if n_num == 0 else numeric_outputs if n_bin == 0 else layers.Concatenate()([binary_outputs, numeric_outputs])
    decoder = models.Model(latent_inputs, outputs_concat, name="decoder")

    # Train
    vae = VAE(encoder, decoder, binary_mask, kl_weight)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    vae.fit(data_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # Generate new samples
    z_new = np.random.normal(size=(n_samples, LATENT_DIM))
    generated_scaled = decoder.predict(z_new, verbose=0)

    # Split binary/numeric sections safely
    gen_bin = generated_scaled[:, :n_bin] if n_bin > 0 else np.zeros((n_samples, 0))
    gen_num = generated_scaled[:, n_bin:] if n_num > 0 else np.zeros((n_samples, 0))

    # Undo scaling by name (in correct order)
    if np.any(numeric_mask):
        gen_num = np.clip(gen_num, 0, 1)
        gen_num = scaler.inverse_transform(gen_num)

    # Reassemble into correct order
    gen_all = np.zeros((n_samples, input_dim), dtype=float)
    gen_all[:, binary_mask] = gen_bin
    gen_all[:, numeric_mask] = gen_num
    generated_df = pd.DataFrame(gen_all, columns=cols)

    # One-hot fix
    for group in one_hot:
        group = [c for c in group if c in generated_df.columns]
        if len(group) == 0:
            continue
        arr = generated_df[group].values
        idx = np.argmax(arr, axis=1)
        arr[:] = 0
        arr[np.arange(len(arr)), idx] = 1
        if arr.shape[1] != len(group):
            raise ValueError(f"Shape mismatch for one-hot group {group}: got {arr.shape}")
        generated_df[group] = arr

    # Round values
    rounding_map = {c: infer_decimals(df[c]) for c in cols}
    for c, n_dec in rounding_map.items():
        if n_dec == 0:
            generated_df[c] = generated_df[c].round().astype(int)
        elif n_dec is not None:
            generated_df[c] = generated_df[c].round(n_dec)

    return generated_df
