import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

class PredictiveRNN(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(PredictiveRNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Define the GRU and Dense layers
        self.gru = tf.keras.layers.GRU(hidden_dim, 
                                      activation='tanh', 
                                      return_sequences=True)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x, training=True):
        # GRU layer
        h = self.gru(x)
        # Output layer
        y_pred = self.dense(h)
        return y_pred

def extract_time(data):
    """Extract time information from the data."""
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i]))
        time.append(len(data[i]))
    return time, max_seq_len

def prepare_prediction_data(data, time, dim, batch_indices):
    """Prepare data for prediction task."""
    X_mb = [data[i][:-1, :(dim-1)] for i in batch_indices]
    T_mb = [time[i]-1 for i in batch_indices]
    Y_mb = [np.reshape(data[i][1:, (dim-1)], 
                      [len(data[i][1:, (dim-1)]), 1]) for i in batch_indices]
    
    # Pad sequences
    max_len = max(len(x) for x in X_mb)
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X_mb, maxlen=max_len, padding='post', dtype='float32')
    Y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        Y_mb, maxlen=max_len, padding='post', dtype='float32')
    
    return tf.convert_to_tensor(X_padded), tf.convert_to_tensor(Y_padded), T_mb

def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        
    Returns:
        - predictive_score: MAE of the predictions on the original data
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])
    
    # Network parameters
    hidden_dim = int(dim/2)
    iterations = 5000
    batch_size = 128
    
    # Create and compile the model
    model = PredictiveRNN(hidden_dim)
    optimizer = tf.keras.optimizers.Adam()
    
    @tf.function
    def train_step(x, y):
        """Single training step."""
        with tf.GradientTape() as tape:
            # Get predictions
            y_pred = model(x, training=True)
            # Calculate loss - using MAE loss
            loss = tf.reduce_mean(tf.abs(y - y_pred))
            
        # Calculate gradients and update model
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    # Training loop using generated data
    for itt in range(iterations):
        # Set mini-batch
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]
        
        # Prepare data
        X_mb, Y_mb, T_mb = prepare_prediction_data(
            generated_data, generated_time, dim, train_idx)
        
        # Train predictor
        step_loss = train_step(X_mb, Y_mb)
        
        if itt % 1000 == 0:
            print(f'Iteration: {itt}, Loss: {step_loss:.4f}')
    
    # Test on original data
    MAE_temp = 0
    num_batches = int(np.ceil(no / batch_size))
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, no)
        test_idx = list(range(start_idx, end_idx))
        
        # Prepare test data
        X_test, Y_test, T_test = prepare_prediction_data(
            ori_data, ori_time, dim, test_idx)
        
        # Get predictions
        pred_Y = model(X_test, training=False)
        
        # Calculate MAE for each sequence
        for j in range(len(test_idx)):
            true_len = T_test[j]
            MAE_temp += mean_absolute_error(
                Y_test[j, :true_len], 
                pred_Y[j, :true_len])
    
    predictive_score = MAE_temp / no
    
    return predictive_score

def normalize_data(data):
    """Normalize the data to [0,1] range."""
    mins = np.min(np.min(data, axis=0), axis=0)
    maxs = np.max(np.max(data, axis=0), axis=0)
    return (data - mins) / (maxs - mins + 1e-7)