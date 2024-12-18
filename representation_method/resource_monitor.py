import psutil
import os
import time
import threading
import GPUtil
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict


class ResourceMonitor:
    """Monitor system resources (CPU, GPU, RAM) during model training."""

    def __init__(self, log_dir: str, interval: float = 1.0):
        """
        Initialize the resource monitor.

        Args:
            log_dir: Directory to save logs and plots
            interval: Monitoring interval in seconds
        """
        self.log_dir = log_dir
        self.interval = interval
        self.is_running = False
        self.monitoring_thread = None
        self.measurements: List[Dict] = []

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(log_dir, 'resource_usage.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)

        # Check for GPU availability
        self.has_gpu = len(GPUtil.getGPUs()) > 0

    def _get_gpu_info(self) -> dict:
        """Get GPU usage information."""
        if not self.has_gpu:
            return {}

        gpus = GPUtil.getGPUs()
        gpu_info = {}

        for i, gpu in enumerate(gpus):
            gpu_info.update({
                f'gpu{i}_usage': gpu.load * 100,
                f'gpu{i}_memory_used': gpu.memoryUsed,
                f'gpu{i}_memory_total': gpu.memoryTotal,
                f'gpu{i}_temperature': gpu.temperature
            })

        return gpu_info

    def _get_system_info(self) -> dict:
        """Get CPU and RAM usage information."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'ram_used': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,  # MB
            'ram_percent': psutil.virtual_memory().percent,
            'timestamp': datetime.now()
        }

    def _monitor(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect system metrics
                metrics = self._get_system_info()

                # Add GPU metrics if available
                if self.has_gpu:
                    metrics.update(self._get_gpu_info())

                # Store measurements
                self.measurements.append(metrics)

                # Log metrics
                log_msg = f"CPU: {metrics['cpu_percent']:.1f}% | RAM: {metrics['ram_used']:.1f}MB ({metrics['ram_percent']:.1f}%)"
                if self.has_gpu:
                    gpu_msg = " | ".join(
                        [f"GPU{i}: {metrics[f'gpu{i}_usage']:.1f}% ({metrics[f'gpu{i}_memory_used']:.0f}MB)"
                         for i in range(len(GPUtil.getGPUs()))])
                    log_msg += f" | {gpu_msg}"

                self.logger.info(log_msg)

                time.sleep(self.interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring: {str(e)}")
                break

    def start(self):
        """Start resource monitoring."""
        if not self.is_running:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitor)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Resource monitoring started")

    def stop(self):
        """Stop resource monitoring and save results."""
        if self.is_running:
            self.is_running = False
            if self.monitoring_thread:
                self.monitoring_thread.join()

            # Save measurements to CSV
            df = pd.DataFrame(self.measurements)
            csv_path = os.path.join(self.log_dir, 'resource_usage.csv')
            df.to_csv(csv_path, index=False)

            # Create plots
            self._create_plots(df)

            self.logger.info("Resource monitoring stopped and data saved")

    def _create_plots(self, df: pd.DataFrame):
        """Create visualization plots of resource usage."""
        # Create timestamp column if not exists
        if 'timestamp' not in df.columns:
            df['timestamp'] = range(len(df))

        # Calculate time in minutes from start
        df['minutes'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60

        # Create plots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # CPU and RAM plot
        ax1 = axes[0]
        ax1.plot(df['minutes'], df['cpu_percent'], label='CPU Usage (%)')
        ax1.plot(df['minutes'], df['ram_percent'], label='RAM Usage (%)')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Usage (%)')
        ax1.set_title('CPU and RAM Usage Over Time')
        ax1.grid(True)
        ax1.legend()

        # GPU plot if available
        if self.has_gpu:
            ax2 = axes[1]
            gpu_cols = [col for col in df.columns if col.startswith('gpu') and col.endswith('usage')]
            for col in gpu_cols:
                gpu_num = col.split('_')[0]
                ax2.plot(df['minutes'], df[col], label=f'{gpu_num.upper()} Usage (%)')

            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Usage (%)')
            ax2.set_title('GPU Usage Over Time')
            ax2.grid(True)
            ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'resource_usage.png'))
        plt.close()


# # Example usage:
# if __name__ == "__main__":
#     # Create monitor instance
#     monitor = ResourceMonitor(log_dir="resource_logs")
#
#     # Start monitoring
#     monitor.start()
#
#     try:
#         # Your training loop would go here
#         for epoch in range(10):
#             time.sleep(5)  # Simulate training
#
#     finally:
#         # Stop monitoring and save results
#         monitor.stop()