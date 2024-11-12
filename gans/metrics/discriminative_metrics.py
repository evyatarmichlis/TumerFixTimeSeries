import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

def train_test_divide(data, data_hat, time, time_hat):
    """Divide train and test data for both original and synthetic data."""
    # Divide train/test for original data
    train_rate = 0.8
    idx = np.random.permutation(len(data))
    train_idx = idx[:int(train_rate * len(data))]
    test_idx = idx[int(train_rate * len(data)):]
    
    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]
    train_time = [time[i] for i in train_idx]
    test_time = [time[i] for i in test_idx]
    
    # Divide train/test for synthetic data
    idx = np.random.permutation(len(data_hat))
    train_idx = idx[:int(train_rate * len(data_hat))]
    test_idx = idx[int(train_rate * len(data_hat)):]
    
    train_data_hat = [data_hat[i] for i in train_idx]
    test_data_hat = [data_hat[i] for i in test_idx]
    train_time_hat = [time_hat[i] for i in train_idx]
    test_time_hat = [time_hat[i] for i in test_idx]
    
    return train_data, train_data_hat, test_data, test_data_hat, \
           train_time, train_time_hat, test_time, test_time_hat

class DiscriminativeRNN(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(DiscriminativeRNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Define the GRU and Dense layers
        self.gru = tf.keras.layers.GRU(hidden_dim, 
                                      activation='tanh', 
                                      return_sequences=False)
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, x, training=True):
        # GRU layer
        h = self.gru(x)
        # Output layer
        y_logit = self.dense(h)
        return y_logit

def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        
    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape    
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
    
    # Network parameters
    hidden_dim = int(dim/2)
    iterations = 2000
    batch_size = 128
    
    # Create and compile the model
    model = DiscriminativeRNN(hidden_dim)
    optimizer = tf.keras.optimizers.Adam()
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
    @tf.function
    def train_step(x_real, x_fake):
        """Single training step."""
        with tf.GradientTape() as tape:
            # Get predictions
            y_logit_real = model(x_real, training=True)
            y_logit_fake = model(x_fake, training=True)
            
            # Calculate losses
            real_loss = bce(tf.ones_like(y_logit_real), y_logit_real)
            fake_loss = bce(tf.zeros_like(y_logit_fake), y_logit_fake)
            total_loss = real_loss + fake_loss
            
        # Calculate gradients and update model
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss
    
    # Training loop
    for itt in range(iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
        
        # Convert to tensors
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        X_hat_mb = tf.convert_to_tensor(X_hat_mb, dtype=tf.float32)
        
        # Train discriminator
        step_loss = train_step(X_mb, X_hat_mb)
        
        if itt % 200 == 0:
            print(f'Iteration: {itt}, Loss: {step_loss:.4f}')
    
    # Test phase
    test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
    test_x_hat = tf.convert_to_tensor(test_x_hat, dtype=tf.float32)
    
    # Get predictions
    y_pred_real = tf.sigmoid(model(test_x, training=False))
    y_pred_fake = tf.sigmoid(model(test_x_hat, training=False))
    
    # Prepare final predictions and labels
    y_pred_final = np.squeeze(np.concatenate((y_pred_real.numpy(), 
                                            y_pred_fake.numpy()), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real),]), 
                                  np.zeros([len(y_pred_fake),])), axis=0)
    
    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score

def extract_time(data):
    """Extract time information from the data."""
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i]))
        time.append(len(data[i]))
    return time, max_seq_len

def batch_generator(data, time, batch_size):
    """Mini-batch generator."""
    idx = np.random.permutation(len(data))
    idx = idx[:batch_size]
    batch_data = [data[i] for i in idx]
    batch_time = [time[i] for i in idx]
    return batch_data, batch_time