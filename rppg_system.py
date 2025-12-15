"""
Real-time Remote Photoplethysmography (rPPG) Heart Rate Detection System
Based on: "Video-based heart rate estimation from challenging scenarios using synthetic video generation"
Benezeth et al., 2024

This implementation includes:
1. Data preprocessing and augmentation
2. Model training (PhysNet/RTrPPG)
3. Real-time heart rate estimation
4. Evaluation metrics
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.fft import fft
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters"""
    # Data parameters
    FRAME_RATE = 30  # fps
    WINDOW_SIZE = 15  # seconds
    STEP_SIZE = 0.5  # seconds
    FRAMES_PER_WINDOW = int(FRAME_RATE * WINDOW_SIZE)
    
    # Heart rate range
    HR_MIN = 40  # bpm
    HR_MAX = 240  # bpm
    FREQ_MIN = HR_MIN / 60.0  # Hz
    FREQ_MAX = HR_MAX / 60.0  # Hz
    
    # Model parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Face detection
    FACE_DETECTOR_PATH = 'models/blazeface.pth'  # Download BlazeFace weights
    
    # Image dimensions
    IMG_SIZE = 128


# ==================== SYNTHETIC PPG SIGNAL GENERATION ====================
class SyntheticPPGGenerator:
    """Generate synthetic PPG signals with realistic characteristics"""
    
    def __init__(self, fps=30, duration=15):
        self.fps = fps
        self.duration = duration
        self.n_samples = int(fps * duration)
        self.t = np.arange(self.n_samples) / fps
        
    def generate_signal(self, hr_mean=75, br_mean=0.25):
        """
        Generate synthetic PPG signal
        
        Args:
            hr_mean: Mean heart rate in bpm
            br_mean: Mean breathing rate in Hz
            
        Returns:
            signal: Synthetic PPG signal
            hr_ref: Reference heart rate
        """
        # Convert HR to Hz
        f_hr = hr_mean / 60.0
        
        # Add variability
        delta_hr = 0.05
        delta_br = 0.1
        
        # Instantaneous heart rate with variability
        f_hr_inst = f_hr + np.random.uniform(-f_hr * delta_hr, 
                                              f_hr * delta_hr, 
                                              self.n_samples)
        
        # Instantaneous breathing rate
        f_br_inst = br_mean + np.random.uniform(-br_mean * delta_br, 
                                                 br_mean * delta_br, 
                                                 self.n_samples)
        
        # Random amplitudes
        A1 = np.random.uniform(0.2, 0.7)  # Pulse amplitude
        A2 = np.random.uniform(0.0, 0.3)  # Dicrotic notch amplitude
        A3 = np.random.uniform(0.3, 2.0)  # Breathing amplitude
        
        # Random phases
        phi_p = np.random.uniform(0, 2 * np.pi)
        phi_b = np.random.uniform(0, 2 * np.pi)
        
        # Constants
        C1 = 0.05
        C2 = 0.01
        C3 = 0.15
        
        # Generate breathing component
        b_t = A3 * np.sin(2 * np.pi * f_br_inst * self.t + phi_b)
        
        # Generate pulse signal
        p_t = (A1 + C2 * b_t) * np.sin(2 * np.pi * (f_hr_inst + C3 * b_t) * self.t + phi_p)
        
        # Generate dicrotic notch
        d_t = (A2 + C2 * b_t) * np.sin(4 * np.pi * (f_hr_inst + C3 * b_t) * self.t + 2 * phi_p)
        
        # Gaussian noise
        n_t = np.random.normal(0, 0.1, self.n_samples)
        
        # Combine components
        s_t = p_t + d_t + C1 * b_t + n_t
        
        # Normalize
        s_t = (s_t - np.mean(s_t)) / (np.std(s_t) + 1e-8)
        
        return s_t, hr_mean


# ==================== DATASET ====================
class rPPGDataset(Dataset):
    """Dataset for rPPG training"""
    
    def __init__(self, video_paths, ppg_signals, hr_labels, transform=None):
        """
        Args:
            video_paths: List of paths to video files
            ppg_signals: List of PPG signals (numpy arrays)
            hr_labels: List of heart rate labels
            transform: Optional transforms
        """
        self.video_paths = video_paths
        self.ppg_signals = ppg_signals
        self.hr_labels = hr_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # Load video frames
        frames = self.load_video(self.video_paths[idx])
        
        # Get PPG signal and HR label
        ppg = self.ppg_signals[idx]
        hr = self.hr_labels[idx]
        
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'frames': torch.FloatTensor(frames),
            'ppg': torch.FloatTensor(ppg),
            'hr': torch.FloatTensor([hr])
        }
    
    def load_video(self, video_path):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while len(frames) < Config.FRAMES_PER_WINDOW:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize and normalize
            frame = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < Config.FRAMES_PER_WINDOW:
            frames.append(frames[-1])
        
        # Convert to (T, H, W, C)
        frames = np.array(frames[:Config.FRAMES_PER_WINDOW])
        
        # Transpose to (T, C, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        
        return frames


# ==================== PHYSNET MODEL ====================
class PhysNet(nn.Module):
    """
    PhysNet: 3D CNN for rPPG estimation
    Based on Yu et al., 2019
    """
    
    def __init__(self, in_channels=3):
        super(PhysNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, 16, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Block 2
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Block 4
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Block 5
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            
            nn.ConvTranspose3d(64, 32, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            
            # Output layer
            nn.Conv3d(32, 1, kernel_size=(1, 1, 1), stride=1, padding=0)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) - Batch of video frames
            
        Returns:
            rppg: (B, T) - Estimated rPPG signal
        """
        # Encode
        x = self.encoder(x)
        
        # Decode
        x = self.decoder(x)
        
        # Spatial pooling
        x = self.adaptive_pool(x)
        
        # Remove spatial dimensions
        rppg = x.squeeze(-1).squeeze(-1).squeeze(1)  # (B, T)
        
        return rppg


# ==================== HEART RATE ESTIMATION ====================
class HeartRateEstimator:
    """Estimate heart rate from rPPG signal"""
    
    @staticmethod
    def estimate_hr(rppg_signal, fps=30):
        """
        Estimate heart rate using FFT
        
        Args:
            rppg_signal: rPPG signal (numpy array)
            fps: Sampling frequency
            
        Returns:
            hr: Estimated heart rate in bpm
            psd: Power spectral density
            freqs: Frequency bins
        """
        # Detrend
        rppg_signal = signal.detrend(rppg_signal)
        
        # Apply Hamming window
        rppg_signal = rppg_signal * np.hamming(len(rppg_signal))
        
        # Compute FFT
        N = len(rppg_signal)
        fft_signal = fft(rppg_signal)
        psd = np.abs(fft_signal) ** 2
        freqs = np.fft.fftfreq(N, 1.0 / fps)
        
        # Keep only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        psd = psd[pos_mask]
        
        # Filter to physiological range
        freq_mask = (freqs >= Config.FREQ_MIN) & (freqs <= Config.FREQ_MAX)
        freqs_filtered = freqs[freq_mask]
        psd_filtered = psd[freq_mask]
        
        # Find peak
        peak_idx = np.argmax(psd_filtered)
        peak_freq = freqs_filtered[peak_idx]
        
        # Convert to bpm
        hr = peak_freq * 60.0
        
        return hr, psd_filtered, freqs_filtered


# ==================== METRICS ====================
class Metrics:
    """Evaluation metrics for rPPG"""
    
    @staticmethod
    def mae(hr_pred, hr_true):
        """Mean Absolute Error"""
        return np.mean(np.abs(hr_pred - hr_true))
    
    @staticmethod
    def pearson_correlation(hr_pred, hr_true):
        """Pearson correlation coefficient"""
        if len(hr_pred) < 2:
            return 0.0
        r, _ = pearsonr(hr_pred, hr_true)
        return r
    
    @staticmethod
    def snr(rppg_signal, hr_true, fps=30):
        """
        Signal-to-Noise Ratio
        
        Args:
            rppg_signal: rPPG signal
            hr_true: True heart rate
            fps: Sampling frequency
            
        Returns:
            snr: SNR in dB
        """
        # Compute FFT
        N = len(rppg_signal)
        fft_signal = fft(rppg_signal)
        psd = np.abs(fft_signal) ** 2
        freqs = np.fft.fftfreq(N, 1.0 / fps)
        
        # Keep positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        psd = psd[pos_mask]
        
        # Filter to physiological range
        freq_mask = (freqs >= Config.FREQ_MIN) & (freqs <= Config.FREQ_MAX)
        freqs_filtered = freqs[freq_mask]
        psd_filtered = psd[freq_mask]
        
        # Find signal frequency
        f_hr = hr_true / 60.0
        
        # Create signal mask (first and second harmonics)
        f_window = 0.1  # Hz
        signal_mask = np.zeros_like(freqs_filtered, dtype=bool)
        
        # First harmonic
        signal_mask |= np.abs(freqs_filtered - f_hr) < f_window
        # Second harmonic
        signal_mask |= np.abs(freqs_filtered - 2 * f_hr) < f_window
        
        # Compute power
        signal_power = np.sum(psd_filtered[signal_mask])
        noise_power = np.sum(psd_filtered[~signal_mask])
        
        # Compute SNR
        if noise_power == 0:
            return 0.0
        
        snr_value = 10 * np.log10(signal_power / noise_power)
        return snr_value
    
    @staticmethod
    def tmc(rppg_signal, fps=30):
        """
        Template Match Correlation
        
        Args:
            rppg_signal: rPPG signal
            fps: Sampling frequency
            
        Returns:
            tmc: TMC value
        """
        # Detect peaks
        peaks, _ = signal.find_peaks(rppg_signal, distance=fps//2)
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate median beat-to-beat interval
        intervals = np.diff(peaks)
        median_interval = int(np.median(intervals))
        
        if median_interval < 10:
            return 0.0
        
        # Extract individual pulses
        pulses = []
        for peak in peaks:
            start = max(0, peak - median_interval // 2)
            end = min(len(rppg_signal), peak + median_interval // 2)
            
            if end - start == median_interval:
                pulse = rppg_signal[start:end]
                pulses.append(pulse)
        
        if len(pulses) < 2:
            return 0.0
        
        # Create template (average pulse)
        template = np.mean(pulses, axis=0)
        
        # Calculate correlation of each pulse with template
        correlations = []
        for pulse in pulses:
            if np.std(pulse) > 0 and np.std(template) > 0:
                corr = np.corrcoef(pulse, template)[0, 1]
                correlations.append(corr)
        
        if len(correlations) == 0:
            return 0.0
        
        # Return mean correlation
        tmc_value = np.mean(correlations)
        return tmc_value


# ==================== TRAINING ====================
class Trainer:
    """Training pipeline for rPPG models"""
    
    def __init__(self, model, train_loader, val_loader, device=Config.DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Metrics tracker
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch in self.train_loader:
            frames = batch['frames'].to(self.device)
            ppg_target = batch['ppg'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            ppg_pred = self.model(frames)
            
            # Compute loss
            loss = self.criterion(ppg_pred, ppg_target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch['frames'].to(self.device)
                ppg_target = batch['ppg'].to(self.device)
                
                # Forward pass
                ppg_pred = self.model(frames)
                
                # Compute loss
                loss = self.criterion(ppg_pred, ppg_target)
                epoch_loss += loss.item()
        
        return epoch_loss / len(self.val_loader)
    
    def train(self, num_epochs=Config.NUM_EPOCHS):
        """Full training loop"""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                logger.info(f"Saved best model with val loss: {val_loss:.4f}")
        
        logger.info("Training completed!")
        
        return self.train_losses, self.val_losses


# ==================== REAL-TIME INFERENCE ====================
class RealTimeRPPG:
    """Real-time rPPG heart rate detection"""
    
    def __init__(self, model_path, device=Config.DEVICE):
        """
        Args:
            model_path: Path to trained model weights
            device: Computation device
        """
        self.device = device
        
        # Load model
        self.model = PhysNet()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Initialize video capture
        self.cap = None
        self.frame_buffer = []
        
        # HR estimator
        self.hr_estimator = HeartRateEstimator()
        
    def start_camera(self, camera_id=0):
        """Start video capture"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FRAME_RATE)
        logger.info("Camera started")
        
    def process_frame(self, frame):
        """Preprocess a single frame"""
        # Resize
        frame = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        frame = np.transpose(frame, (2, 0, 1))
        
        return frame
    
    def estimate_hr_realtime(self):
        """Run real-time heart rate estimation"""
        if self.cap is None:
            self.start_camera()
        
        logger.info("Starting real-time heart rate estimation...")
        logger.info("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            self.frame_buffer.append(processed_frame)
            
            # Keep buffer size
            if len(self.frame_buffer) > Config.FRAMES_PER_WINDOW:
                self.frame_buffer.pop(0)
            
            # Estimate HR when buffer is full
            if len(self.frame_buffer) == Config.FRAMES_PER_WINDOW:
                # Prepare input
                frames = np.array(self.frame_buffer)
                frames = torch.FloatTensor(frames).unsqueeze(0)  # (1, T, C, H, W)
                frames = frames.transpose(1, 2)  # (1, C, T, H, W)
                frames = frames.to(self.device)
                
                # Predict rPPG
                with torch.no_grad():
                    rppg_pred = self.model(frames)
                    rppg_pred = rppg_pred.cpu().numpy().squeeze()
                
                # Estimate HR
                hr, _, _ = self.hr_estimator.estimate_hr(rppg_pred, Config.FRAME_RATE)
                
                # Display
                cv2.putText(frame, f"HR: {hr:.1f} bpm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Real-time rPPG', frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time estimation stopped")


# ==================== MAIN ====================
def main():
    """Main function demonstrating usage"""
    
    # Example 1: Generate synthetic PPG signals
    logger.info("Generating synthetic PPG signals...")
    ppg_gen = SyntheticPPGGenerator(fps=Config.FRAME_RATE, duration=Config.WINDOW_SIZE)
    signal, hr = ppg_gen.generate_signal(hr_mean=75)
    
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(f"Synthetic PPG Signal (HR: {hr} bpm)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig('synthetic_ppg.png')
    logger.info("Synthetic PPG signal saved to synthetic_ppg.png")
    
    # Example 2: Create model
    logger.info("Creating PhysNet model...")
    model = PhysNet(in_channels=3)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example 3: Heart rate estimation from signal
    logger.info("Estimating heart rate from signal...")
    hr_est = HeartRateEstimator()
    hr_estimated, psd, freqs = hr_est.estimate_hr(signal, Config.FRAME_RATE)
    logger.info(f"Estimated HR: {hr_estimated:.2f} bpm (True: {hr} bpm)")
    
    # Example 4: Compute metrics
    logger.info("Computing metrics...")
    metrics = Metrics()
    snr = metrics.snr(signal, hr, Config.FRAME_RATE)
    tmc = metrics.tmc(signal, Config.FRAME_RATE)
    logger.info(f"SNR: {snr:.2f} dB, TMC: {tmc:.2f}")
    
    logger.info("\nSetup complete! You can now:")
    logger.info("1. Prepare your dataset using rPPGDataset")
    logger.info("2. Train the model using Trainer")
    logger.info("3. Run real-time inference using RealTimeRPPG")


if __name__ == "__main__":
    main()