import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import json
from datetime import datetime


def extract_time(data):
    """Extract time information from the data."""
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i]))
        time.append(len(data[i]))
    return time, max_seq_len


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation."""
    return np.random.uniform(0., 1., size=(batch_size, max_seq_len, z_dim))


def batch_generator(data, time, batch_size):
    """Mini-batch generator."""
    idx = np.random.permutation(len(data))
    idx = idx[:batch_size]
    batch_data = [data[i] for i in idx]
    batch_time = [time[i] for i in idx]
    return batch_data, batch_time


def create_rnn_layer(hidden_dim, num_layers, module_name='gru'):
    """Create stacked RNN layers."""
    if module_name == 'gru':
        rnn_cells = [layers.GRUCell(hidden_dim) for _ in range(num_layers)]
    elif module_name == 'lstm':
        rnn_cells = [layers.LSTMCell(hidden_dim) for _ in range(num_layers)]
    else:
        rnn_cells = [layers.SimpleRNNCell(hidden_dim) for _ in range(num_layers)]
    return layers.RNN(rnn_cells, return_sequences=True)


class TimeGAN(tf.keras.Model):
    def __init__(self, parameters):
        super(TimeGAN, self).__init__()
        self.hidden_dim = parameters['hidden_dim']
        self.num_layers = parameters['num_layers']
        self.module_name = parameters.get('module_name', 'gru')
        self.gamma = 1
        self.parameters = parameters
        self.min_val = None
        self.max_val = None
        self.training_history = {
            'embedding_losses': [],
            'supervisor_losses': [],
            'generator_losses': [],
            'discriminator_losses': [],
            'iterations': []
        }

        # Create network components
        self.embedder = tf.keras.Sequential([
            create_rnn_layer(self.hidden_dim, self.num_layers, self.module_name),
            layers.Dense(self.hidden_dim, activation='sigmoid')
        ])

        self.recovery = tf.keras.Sequential([
            create_rnn_layer(self.hidden_dim, self.num_layers, self.module_name),
            layers.Dense(parameters['feature_dim'], activation='sigmoid')
        ])

        self.generator = tf.keras.Sequential([
            create_rnn_layer(self.hidden_dim, self.num_layers, self.module_name),
            layers.Dense(self.hidden_dim, activation='sigmoid')
        ])

        self.supervisor = tf.keras.Sequential([
            create_rnn_layer(self.hidden_dim, self.num_layers - 1, self.module_name),
            layers.Dense(self.hidden_dim, activation='sigmoid')
        ])

        self.discriminator = tf.keras.Sequential([
            create_rnn_layer(self.hidden_dim, self.num_layers, self.module_name),
            layers.Dense(1)
        ])

    def compile(self):
        super(TimeGAN, self).compile()
        self.e_optimizer = tf.keras.optimizers.Adam()
        self.r_optimizer = tf.keras.optimizers.Adam()
        self.g_optimizer = tf.keras.optimizers.Adam()
        self.s_optimizer = tf.keras.optimizers.Adam()
        self.d_optimizer = tf.keras.optimizers.Adam()

    def save_model(self, save_dir):
        """Save the TimeGAN model and its components."""
        try:
            os.makedirs(save_dir, exist_ok=True)

            # Save model weights
            self.embedder.save_weights(os.path.join(save_dir, 'embedder.weights.h5'))
            self.recovery.save_weights(os.path.join(save_dir, 'recovery.weights.h5'))
            self.generator.save_weights(os.path.join(save_dir, 'generator.weights.h5'))
            self.supervisor.save_weights(os.path.join(save_dir, 'supervisor.weights.h5'))
            self.discriminator.save_weights(os.path.join(save_dir, 'discriminator.weights.h5'))

            # For TensorFlow 2.x, we'll save the optimizer configurations instead of weights
            optimizer_config = {
                'e_optimizer': self.e_optimizer.get_config(),
                'r_optimizer': self.r_optimizer.get_config(),
                'g_optimizer': self.g_optimizer.get_config(),
                's_optimizer': self.s_optimizer.get_config(),
                'd_optimizer': self.d_optimizer.get_config()
            }

            # Save parameters, training history, and optimizer configs
            metadata = {
                'parameters': self.parameters,
                'training_history': self.training_history,
                'normalization': {
                    'min_val': self.min_val.tolist() if self.min_val is not None else None,
                    'max_val': self.max_val.tolist() if self.max_val is not None else None
                },
                'optimizer_config': optimizer_config,
                'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)

            print(f"Model saved successfully to {save_dir}")

        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, load_dir):
        """Load the TimeGAN model and its components."""
        try:
            # Load metadata
            with open(os.path.join(load_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)

            # Load normalization parameters
            self.min_val = np.array(metadata['normalization']['min_val'])
            self.max_val = np.array(metadata['normalization']['max_val'])
            self.training_history = metadata['training_history']

            # Load model weights
            self.embedder.load_weights(os.path.join(load_dir, 'embedder.weights.h5'))
            self.recovery.load_weights(os.path.join(load_dir, 'recovery.weights.h5'))
            self.generator.load_weights(os.path.join(load_dir, 'generator.weights.h5'))
            self.supervisor.load_weights(os.path.join(load_dir, 'supervisor.weights.h5'))
            self.discriminator.load_weights(os.path.join(load_dir, 'discriminator.weights.h5'))

            # Recreate optimizers with saved configurations
            optimizer_config = metadata['optimizer_config']
            self.e_optimizer = tf.keras.optimizers.Adam.from_config(optimizer_config['e_optimizer'])
            self.r_optimizer = tf.keras.optimizers.Adam.from_config(optimizer_config['r_optimizer'])
            self.g_optimizer = tf.keras.optimizers.Adam.from_config(optimizer_config['g_optimizer'])
            self.s_optimizer = tf.keras.optimizers.Adam.from_config(optimizer_config['s_optimizer'])
            self.d_optimizer = tf.keras.optimizers.Adam.from_config(optimizer_config['d_optimizer'])

            print(f"Model loaded successfully from {load_dir}")
            print(f"Model was saved at: {metadata.get('save_time', 'time not recorded')}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    @tf.function
    def train_embedder(self, X, T):
        """Train embedder network."""
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = tf.reduce_mean(tf.square(X - X_tilde))
            E_loss = 10 * tf.sqrt(E_loss_T0)

        gradients = tape.gradient(E_loss,
                                  self.embedder.trainable_variables +
                                  self.recovery.trainable_variables)
        self.e_optimizer.apply_gradients(
            zip(gradients,
                self.embedder.trainable_variables +
                self.recovery.trainable_variables))
        return E_loss

    @tf.function
    def train_supervisor(self, X, Z, T):
        """Train with supervised loss only."""
        with tf.GradientTape() as tape:
            E_hat = self.generator(Z, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            H = self.embedder(X, training=True)
            G_loss_S = tf.reduce_mean(tf.square(H[:, 1:, :] - H_hat[:, :-1, :]))

        gradients = tape.gradient(G_loss_S,
                                  self.generator.trainable_variables +
                                  self.supervisor.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gradients,
                self.generator.trainable_variables +
                self.supervisor.trainable_variables))
        return G_loss_S

    @tf.function
    def train_generator(self, X, Z, T):
        """Train generator network."""
        with tf.GradientTape() as tape:
            E_hat = self.generator(Z, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            H = self.embedder(X, training=True)

            Y_fake = self.discriminator(H_hat, training=True)
            Y_fake_e = self.discriminator(E_hat, training=True)

            G_loss_U = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(Y_fake), Y_fake, from_logits=True))
            G_loss_U_e = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(Y_fake_e), Y_fake_e, from_logits=True))

            G_loss_S = tf.reduce_mean(tf.square(H[:, 1:, :] - H_hat[:, :-1, :]))

            X_hat = self.recovery(H_hat, training=True)
            G_loss_V1 = tf.reduce_mean(tf.abs(
                tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) -
                tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
            G_loss_V2 = tf.reduce_mean(tf.abs(
                tf.nn.moments(X_hat, [0])[0] - tf.nn.moments(X, [0])[0]))

            G_loss_V = G_loss_V1 + G_loss_V2

            G_loss = (G_loss_U +
                      self.gamma * G_loss_U_e +
                      100 * tf.sqrt(G_loss_S) +
                      100 * G_loss_V)

        gradients = tape.gradient(G_loss,
                                  self.generator.trainable_variables +
                                  self.supervisor.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gradients,
                self.generator.trainable_variables +
                self.supervisor.trainable_variables))
        return G_loss, G_loss_U, G_loss_S, G_loss_V

    @tf.function
    def train_discriminator(self, X, Z, T):
        """Train discriminator network."""
        with tf.GradientTape() as tape:
            E_hat = self.generator(Z, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            H = self.embedder(X, training=True)

            Y_fake = self.discriminator(H_hat, training=True)
            Y_real = self.discriminator(H, training=True)
            Y_fake_e = self.discriminator(E_hat, training=True)

            D_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(Y_real), Y_real, from_logits=True))
            D_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(Y_fake), Y_fake, from_logits=True))
            D_loss_fake_e = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(Y_fake_e), Y_fake_e, from_logits=True))

            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e

        gradients = tape.gradient(D_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables))
        return D_loss

    def generate_samples(self, n_samples, ori_time, max_seq_len):
        """Generate synthetic samples using the trained model."""
        z_dim = self.parameters['feature_dim']

        # Generate random noise
        Z_mb = random_generator(n_samples, z_dim, ori_time, max_seq_len)
        Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        # Generate synthetic data
        E_hat = self.generator(Z_mb)
        H_hat = self.supervisor(E_hat)
        generated_data_raw = self.recovery(H_hat).numpy()

        # Post-processing
        generated_data = []
        for i in range(n_samples):
            # Extract sequence up to original length
            temp = generated_data_raw[i, :ori_time[i], :]
            # Denormalize
            if self.min_val is not None and self.max_val is not None:
                temp = temp * (self.max_val - self.min_val)
                temp = temp + self.min_val
            generated_data.append(temp)

        return generated_data



def timegan(ori_data, parameters):
    """Main TimeGAN function."""
    # Initialize parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    parameters['feature_dim'] = dim

    # Create TimeGAN instance
    model = TimeGAN(parameters)
    model.compile()

    # Time information extraction
    ori_time, max_seq_len = extract_time(ori_data)

    # Store normalization parameters
    model.min_val = np.min(np.min(ori_data, axis=0), axis=0)
    model.max_val = np.max(np.max(ori_data, axis=0), axis=0)

    # Normalization
    ori_data = (ori_data - model.min_val) / (model.max_val - model.min_val + 1e-7)

    # Training parameters
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim = dim

    print('Start Embedding Network Training')
    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)

        # Train embedder
        step_e_loss = model.train_embedder(X_mb, T_mb)

        # Record history and save checkpoint
        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, e_loss: {np.round(np.sqrt(step_e_loss), 4)}')
            model.training_history['embedding_losses'].append(float(np.sqrt(step_e_loss)))
            model.training_history['iterations'].append(itt)

            # Save checkpoint if directory specified
            if parameters.get('save_dir'):
                checkpoint_dir = os.path.join(parameters['save_dir'], f'embedding_checkpoint_{itt}')
                model.save_model(checkpoint_dir)

    print('Start Training with Supervised Loss Only')
    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        # Train supervisor
        step_g_loss_s = model.train_supervisor(X_mb, Z_mb, T_mb)

        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(step_g_loss_s), 4)}')
            model.training_history['supervisor_losses'].append(float(np.sqrt(step_g_loss_s)))

            # Save checkpoint if directory specified
            if parameters.get('save_dir'):
                checkpoint_dir = os.path.join(parameters['save_dir'], f'supervisor_checkpoint_{itt}')
                model.save_model(checkpoint_dir)

    print('Start Joint Training')
    for itt in range(iterations):
        # Generator training (2 times per iteration)
        for kk in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

            # Train generator
            step_g_loss, step_g_loss_u, step_g_loss_s, step_g_loss_v = model.train_generator(X_mb, Z_mb, T_mb)
            step_e_loss = model.train_embedder(X_mb, T_mb)

        # Discriminator training
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        step_d_loss = model.train_discriminator(X_mb, Z_mb, T_mb)

        # Record history and save checkpoint
        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, d_loss: {np.round(step_d_loss, 4)}, '
                  f'g_loss_u: {np.round(step_g_loss_u, 4)}, '
                  f'g_loss_s: {np.round(np.sqrt(step_g_loss_s), 4)}, '
                  f'g_loss_v: {np.round(step_g_loss_v, 4)}, '
                  f'e_loss: {np.round(np.sqrt(step_e_loss), 4)}')

            model.training_history['discriminator_losses'].append(float(step_d_loss))
            model.training_history['generator_losses'].append(float(step_g_loss))

            # Save checkpoint if directory specified
            if parameters.get('save_dir'):
                checkpoint_dir = os.path.join(parameters['save_dir'], f'joint_checkpoint_{itt}')
                model.save_model(checkpoint_dir)

    # Save final model if directory specified
    if parameters.get('save_dir'):
        final_dir = os.path.join(parameters['save_dir'], 'final_model')
        model.save_model(final_dir)

    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = model.generate_samples(no, ori_time, max_seq_len)

    return synthetic_data


class EyeTrackingTimeGAN(TimeGAN):
    def __init__(self, parameters):
        super().__init__(parameters)

        # Define feature weights for eye tracking data - will be used in the recovery phase
        self.feature_weights = tf.constant([
            2.0,  # Pupil_Size - highest importance
            3.0,  # CURRENT_FIX_DURATION - second highest
            # 1.0,  # CURRENT_FIX_IA_X
            # 1.0,  # CURRENT_FIX_IA_Y
            # 1.0,  # CURRENT_FIX_INDEX
            # 1.0, # CURRENT_FIX_COMPONENT_COUNT
            5.0,  # Pupil_Size_Max_Diff
            5.0  # CURRENT_FIX_DURATION_Max_Diff
        ], dtype=tf.float32)

        # Expand dimensions for broadcasting
        self.feature_weights = tf.reshape(self.feature_weights, (1, 1, -1))

        # Create uniform weights for hidden states (since they're embeddings)
        self.hidden_weights = tf.ones((1, 1, self.hidden_dim), dtype=tf.float32)

    @tf.function
    def weighted_mse_loss(self, y_true, y_pred, is_hidden=False):
        """Weighted MSE loss that emphasizes important features"""
        squared_diff = tf.square(y_true - y_pred)
        if is_hidden:
            # For hidden states, use uniform weights
            weighted_diff = squared_diff * self.hidden_weights
        else:
            # For features, use feature importance weights
            weighted_diff = squared_diff * self.feature_weights
        return tf.reduce_mean(weighted_diff)

    @tf.function
    def train_supervisor(self, X, Z, T):
        """Modified supervisor training with weighted loss"""
        with tf.GradientTape() as tape:
            E_hat = self.generator(Z, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            H = self.embedder(X, training=True)
            G_loss_S = self.weighted_mse_loss(H[:, 1:, :], H_hat[:, :-1, :], is_hidden=True)

        gradients = tape.gradient(G_loss_S,
                              self.generator.trainable_variables +
                              self.supervisor.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gradients,
                self.generator.trainable_variables +
                self.supervisor.trainable_variables))
        return G_loss_S

    @tf.function
    def train_embedder(self, X, T):
        """Modified embedder training with weighted loss"""
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)

            E_loss_T0 = self.weighted_mse_loss(X, X_tilde, is_hidden=False)
            E_loss = 10 * tf.sqrt(E_loss_T0)

        gradients = tape.gradient(E_loss,
                                  self.embedder.trainable_variables +
                                  self.recovery.trainable_variables)
        self.e_optimizer.apply_gradients(
            zip(gradients,
                self.embedder.trainable_variables +
                self.recovery.trainable_variables))
        return E_loss

    @tf.function
    def train_generator(self, X, Z, T):
        """Modified generator training with weighted reconstruction"""
        with tf.GradientTape() as tape:
            E_hat = self.generator(Z, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            H = self.embedder(X, training=True)

            Y_fake = self.discriminator(H_hat, training=True)
            Y_fake_e = self.discriminator(E_hat, training=True)

            G_loss_U = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(Y_fake), Y_fake, from_logits=True))
            G_loss_U_e = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(Y_fake_e), Y_fake_e, from_logits=True))

            G_loss_S = self.weighted_mse_loss(H[:, 1:, :], H_hat[:, :-1, :], is_hidden=True)

            X_hat = self.recovery(H_hat, training=True)

            X_weighted = X * self.feature_weights
            X_hat_weighted = X_hat * self.feature_weights

            G_loss_V1 = tf.reduce_mean(tf.abs(
                tf.sqrt(tf.nn.moments(X_hat_weighted, [0])[1] + 1e-6) -
                tf.sqrt(tf.nn.moments(X_weighted, [0])[1] + 1e-6)))
            G_loss_V2 = tf.reduce_mean(tf.abs(
                tf.nn.moments(X_hat_weighted, [0])[0] -
                tf.nn.moments(X_weighted, [0])[0]))

            G_loss_V = G_loss_V1 + G_loss_V2

            G_loss = (G_loss_U +
                      self.gamma * G_loss_U_e +
                      100 * tf.sqrt(G_loss_S) +
                      100 * G_loss_V)

        gradients = tape.gradient(G_loss,
                                  self.generator.trainable_variables +
                                  self.supervisor.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gradients,
                self.generator.trainable_variables +
                self.supervisor.trainable_variables))
        return G_loss, G_loss_U, G_loss_S, G_loss_V

    def generate_samples(self, n_samples, ori_time, max_seq_len):
        """Generate synthetic samples using the trained model."""
        z_dim = self.parameters['feature_dim']
        batch_size = 1000  # Generate in batches to avoid memory issues
        generated_data = []
        remaining_samples = n_samples

        while remaining_samples > 0:
            # Determine batch size for this iteration
            current_batch = min(batch_size, remaining_samples)

            # Generate random noise
            Z_mb = random_generator(current_batch, z_dim, ori_time[:current_batch], max_seq_len)
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

            # Generate synthetic data
            E_hat = self.generator(Z_mb)
            H_hat = self.supervisor(E_hat)
            generated_data_raw = self.recovery(H_hat).numpy()

            # Process each sample in the batch
            for i in range(current_batch):
                # Use median sequence length from original data for consistency
                median_len = int(np.median(ori_time))
                temp = generated_data_raw[i, :median_len, :]

                # Denormalize
                if self.min_val is not None and self.max_val is not None:
                    temp = temp * (self.max_val - self.min_val)
                    temp = temp + self.min_val

                generated_data.append(temp)

            remaining_samples -= current_batch
            print(f"Generated {len(generated_data)} samples out of {n_samples} needed")

        return generated_data

def weighted_timegan(ori_data, parameters):
    """Modified TimeGAN function using weighted loss for eye tracking data"""
    # Initialize parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    no  = parameters["no"]
    parameters['feature_dim'] = dim

    # Create EyeTrackingTimeGAN instance instead of regular TimeGAN
    model = EyeTrackingTimeGAN(parameters)
    model.compile()

    ori_time, max_seq_len = extract_time(ori_data)


    if parameters.get('load_dir') and os.path.exists(parameters['load_dir']):
        print(f"Loading existing model from {parameters['load_dir']}")
        if model.load_model(parameters['load_dir']):
            print("Model loaded successfully")
            # Generate synthetic data using loaded model
            synthetic_data = model.generate_samples(no, ori_time, max_seq_len)
            return synthetic_data
        else:
            print("Failed to load model, falling back to training")


    # Rest of the training process remains the same as original timegan function
    # but uses the weighted loss implementations

    model.min_val = np.min(np.min(ori_data, axis=0), axis=0)
    model.max_val = np.max(np.max(ori_data, axis=0), axis=0)
    ori_data = (ori_data - model.min_val) / (model.max_val - model.min_val + 1e-7)

    # Training parameters
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim = dim

    print('Start Embedding Network Training with Feature Weighting')
    # ... rest of training code same as original timegan ...

    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)

        # Train embedder
        step_e_loss = model.train_embedder(X_mb, T_mb)

        # Record history and save checkpoint
        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, e_loss: {np.round(np.sqrt(step_e_loss), 4)}')
            model.training_history['embedding_losses'].append(float(np.sqrt(step_e_loss)))
            model.training_history['iterations'].append(itt)

            # Save checkpoint if directory specified
            if parameters.get('save_dir'):
                checkpoint_dir = os.path.join(parameters['save_dir'], f'embedding_checkpoint_{itt}')
                model.save_model(checkpoint_dir)

    print('Start Training with Supervised Loss Only')
    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        # Train supervisor
        step_g_loss_s = model.train_supervisor(X_mb, Z_mb, T_mb)

        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(step_g_loss_s), 4)}')
            model.training_history['supervisor_losses'].append(float(np.sqrt(step_g_loss_s)))

            # Save checkpoint if directory specified
            if parameters.get('save_dir'):
                checkpoint_dir = os.path.join(parameters['save_dir'], f'supervisor_checkpoint_{itt}')
                model.save_model(checkpoint_dir)

    print('Start Joint Training')
    for itt in range(iterations):
        # Generator training (2 times per iteration)
        for kk in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

            # Train generator
            step_g_loss, step_g_loss_u, step_g_loss_s, step_g_loss_v = model.train_generator(X_mb, Z_mb, T_mb)
            step_e_loss = model.train_embedder(X_mb, T_mb)

        # Discriminator training
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        step_d_loss = model.train_discriminator(X_mb, Z_mb, T_mb)

        # Record history and save checkpoint
        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, d_loss: {np.round(step_d_loss, 4)}, '
                  f'g_loss_u: {np.round(step_g_loss_u, 4)}, '
                  f'g_loss_s: {np.round(np.sqrt(step_g_loss_s), 4)}, '
                  f'g_loss_v: {np.round(step_g_loss_v, 4)}, '
                  f'e_loss: {np.round(np.sqrt(step_e_loss), 4)}')

            model.training_history['discriminator_losses'].append(float(step_d_loss))
            model.training_history['generator_losses'].append(float(step_g_loss))

            # Save checkpoint if directory specified
            if parameters.get('save_dir'):
                checkpoint_dir = os.path.join(parameters['save_dir'], f'joint_checkpoint_{itt}')
                model.save_model(checkpoint_dir)

    # Save final model if directory specified
    if parameters.get('save_dir'):
        final_dir = os.path.join(parameters['save_dir'], 'final_model')
        model.save_model(final_dir)

    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = model.generate_samples(no, ori_time, max_seq_len)

    return synthetic_data