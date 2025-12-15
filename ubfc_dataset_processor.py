"""
UBFC-rPPG Dataset Processor
Prepares UBFC-rPPG datasets for training with our rPPG system

Citation:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, 
Unsupervised skin tissue segmentation for remote photoplethysmography, 
Pattern Recognition Letters, Elsevier, 2017.

Dataset structure:
DATASET_1/
├── subject1/
│   ├── vid.avi
│   └── gtdump.xmp
├── subject2/
│   └── ...

DATASET_2/
├── subject1/
│   ├── vid.avi
│   └── ground_truth.txt
├── subject2/
│   └── ...
"""

import os
import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UBFCDatasetProcessor:
    """Process UBFC-rPPG dataset and prepare for training"""
    
    def __init__(self, root_folder: str, dataset_type: str = 'DATASET_2'):
        """
        Args:
            root_folder: Path to dataset root (e.g., 'DATASET_2/')
            dataset_type: 'DATASET_1' or 'DATASET_2'
        """
        self.root_folder = Path(root_folder)
        self.dataset_type = dataset_type
        
        if not self.root_folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {root_folder}")
        
        logger.info(f"Initialized processor for {dataset_type} at {root_folder}")
    
    def load_ground_truth(self, subject_folder: Path) -> Tuple[Optional[np.ndarray], 
                                                                 Optional[np.ndarray], 
                                                                 Optional[np.ndarray]]:
        """
        Load ground truth PPG signal from subject folder
        
        Args:
            subject_folder: Path to subject folder
            
        Returns:
            gt_trace: PPG signal (normalized)
            gt_time: Time steps in seconds
            gt_hr: Heart rate values from sensor
        """
        gt_trace = None
        gt_time = None
        gt_hr = None
        
        # Try DATASET_1 format first
        gt_file_1 = subject_folder / 'gtdump.xmp'
        if gt_file_1.exists():
            try:
                gt_data = pd.read_csv(gt_file_1, header=None).values
                gt_trace = gt_data[:, 3]  # 4th column
                gt_time = gt_data[:, 0] / 1000  # 1st column, convert to seconds
                gt_hr = gt_data[:, 1]  # 2nd column
                logger.debug(f"Loaded ground truth (DATASET_1 format) from {gt_file_1}")
            except Exception as e:
                logger.error(f"Error reading {gt_file_1}: {e}")
        
        # Try DATASET_2 format
        gt_file_2 = subject_folder / 'ground_truth.txt'
        if gt_file_2.exists() and gt_trace is None:
            try:
                gt_data = np.loadtxt(gt_file_2)
                gt_trace = gt_data[0, :]  # 1st row
                gt_time = gt_data[2, :]  # 3rd row
                gt_hr = gt_data[1, :]  # 2nd row
                logger.debug(f"Loaded ground truth (DATASET_2 format) from {gt_file_2}")
            except Exception as e:
                logger.error(f"Error reading {gt_file_2}: {e}")
        
        if gt_trace is None:
            logger.warning(f"No ground truth found in {subject_folder}")
            return None, None, None
        
        # Normalize (zero mean, unit variance)
        gt_trace = gt_trace - np.mean(gt_trace)
        if np.std(gt_trace) > 0:
            gt_trace = gt_trace / np.std(gt_trace)
        else:
            logger.warning("Standard deviation is zero, normalization skipped")
        
        return gt_trace, gt_time, gt_hr
    
    def calculate_heart_rate_from_ppg(self, ppg_signal: np.ndarray, 
                                       fps: float,
                                       window_size: int = 450,
                                       step_size: int = 15) -> Tuple[float, List[float]]:
        """
        Calculate heart rate from PPG signal using FFT
        
        Args:
            ppg_signal: PPG signal
            fps: Sampling frequency (frame rate)
            window_size: Window size for HR calculation
            step_size: Step size for sliding window
            
        Returns:
            mean_hr: Mean heart rate
            hr_values: List of HR values from sliding window
        """
        if len(ppg_signal) < window_size:
            # Use entire signal
            hr = self._estimate_hr_single_window(ppg_signal, fps)
            return hr, [hr]
        
        # Sliding window approach
        hr_values = []
        for start_idx in range(0, len(ppg_signal) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = ppg_signal[start_idx:end_idx]
            hr = self._estimate_hr_single_window(window, fps)
            if hr > 0:
                hr_values.append(hr)
        
        mean_hr = np.mean(hr_values) if hr_values else 0.0
        return mean_hr, hr_values
    
    def _estimate_hr_single_window(self, signal_window: np.ndarray, fps: float) -> float:
        """Estimate HR from a single window using FFT"""
        # Detrend
        signal_window = signal.detrend(signal_window)
        
        # Apply Hamming window
        signal_window = signal_window * np.hamming(len(signal_window))
        
        # FFT
        N = len(signal_window)
        fft_signal = fft(signal_window)
        psd = np.abs(fft_signal) ** 2
        freqs = np.fft.fftfreq(N, 1.0 / fps)
        
        # Keep positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        psd = psd[pos_mask]
        
        # Filter to physiological range (0.7-3 Hz = 42-180 bpm)
        freq_mask = (freqs >= 0.7) & (freqs <= 3.0)
        freqs_filtered = freqs[freq_mask]
        psd_filtered = psd[freq_mask]
        
        if len(psd_filtered) == 0:
            return 0.0
        
        # Find peak
        peak_idx = np.argmax(psd_filtered)
        peak_freq = freqs_filtered[peak_idx]
        hr = peak_freq * 60.0
        
        return hr
    
    def extract_video_info(self, video_path: Path) -> Dict:
        """Extract video metadata"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration
        }
    
    def process_subject(self, subject_folder: Path, 
                        visualize: bool = False) -> Optional[Dict]:
        """
        Process a single subject's data
        
        Args:
            subject_folder: Path to subject folder
            visualize: Whether to show plots
            
        Returns:
            subject_data: Dict with processed data
        """
        subject_name = subject_folder.name
        logger.info(f"Processing subject: {subject_name}")
        
        # Load ground truth
        gt_trace, gt_time, gt_hr = self.load_ground_truth(subject_folder)
        
        if gt_trace is None:
            logger.error(f"Failed to load ground truth for {subject_name}")
            return None
        
        # Check video
        video_path = subject_folder / 'vid.avi'
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None
        
        # Extract video info
        video_info = self.extract_video_info(video_path)
        if video_info is None:
            return None
        
        # Calculate heart rate from PPG signal
        mean_hr, hr_values = self.calculate_heart_rate_from_ppg(
            gt_trace, 
            video_info['fps']
        )
        
        # Prepare data
        subject_data = {
            'subject_name': subject_name,
            'video_path': str(video_path),
            'ppg_signal': gt_trace.tolist(),
            'ppg_time': gt_time.tolist() if gt_time is not None else None,
            'heart_rate': float(mean_hr),
            'hr_values': [float(hr) for hr in hr_values],
            'video_info': video_info,
            'ppg_length': len(gt_trace),
            'duration': video_info['duration']
        }
        
        logger.info(f"  Video: {video_info['total_frames']} frames @ {video_info['fps']:.2f} fps")
        logger.info(f"  Duration: {video_info['duration']:.2f} seconds")
        logger.info(f"  PPG signal length: {len(gt_trace)} samples")
        logger.info(f"  Mean heart rate: {mean_hr:.2f} bpm")
        logger.info(f"  HR range: {min(hr_values):.1f} - {max(hr_values):.1f} bpm")
        
        # Visualize if requested
        if visualize and gt_time is not None:
            self._visualize_subject_data(subject_data, gt_trace, gt_time, hr_values)
        
        return subject_data
    
    def _visualize_subject_data(self, subject_data: Dict, 
                                 gt_trace: np.ndarray,
                                 gt_time: np.ndarray,
                                 hr_values: List[float]):
        """Create visualization for subject data"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot PPG signal
        axes[0].plot(gt_time, gt_trace)
        axes[0].set_title(f'PPG Signal - {subject_data["subject_name"]}')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Normalized Amplitude')
        axes[0].grid(True)
        
        # Plot heart rate values
        axes[1].plot(hr_values, marker='o')
        axes[1].axhline(y=subject_data['heart_rate'], color='r', 
                       linestyle='--', label=f'Mean: {subject_data["heart_rate"]:.1f} bpm')
        axes[1].set_title('Heart Rate Estimates')
        axes[1].set_xlabel('Window Index')
        axes[1].set_ylabel('Heart Rate (bpm)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def process_all_subjects(self, visualize: bool = False, 
                            output_json: str = 'annotations.json') -> Dict:
        """
        Process all subjects in the dataset
        
        Args:
            visualize: Whether to show plots for each subject
            output_json: Path to save annotations
            
        Returns:
            annotations: Dict with all subject data
        """
        # Get all subject folders
        subject_folders = [
            f for f in self.root_folder.iterdir() 
            if f.is_dir() and f.name not in ['.', '..', 'desktop.ini', '__pycache__']
        ]
        subject_folders.sort()
        
        if not subject_folders:
            logger.error(f"No subject folders found in {self.root_folder}")
            return {}
        
        logger.info(f"Found {len(subject_folders)} subjects")
        
        annotations = {}
        successful = 0
        failed = 0
        
        for subject_folder in tqdm(subject_folders, desc="Processing subjects"):
            subject_data = self.process_subject(subject_folder, visualize=visualize)
            
            if subject_data is not None:
                # Store with video filename as key (without extension)
                key = subject_data['subject_name']
                annotations[key] = {
                    'ppg': subject_data['ppg_signal'],
                    'heart_rate': subject_data['heart_rate'],
                    'video_path': subject_data['video_path'],
                    'video_info': subject_data['video_info'],
                    'hr_values': subject_data['hr_values']
                }
                successful += 1
            else:
                failed += 1
        
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        
        # Save annotations
        if annotations:
            with open(output_json, 'w') as f:
                json.dump(annotations, f, indent=2)
            logger.info(f"\nAnnotations saved to: {output_json}")
        
        return annotations
    
    def create_dataset_summary(self, annotations: Dict, 
                               output_file: str = 'dataset_summary.txt'):
        """Create a summary report of the dataset"""
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"UBFC-rPPG Dataset Summary - {self.dataset_type}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total subjects: {len(annotations)}\n\n")
            
            # Calculate statistics
            all_hrs = [data['heart_rate'] for data in annotations.values()]
            all_durations = [data['video_info']['duration'] for data in annotations.values()]
            
            f.write("Heart Rate Statistics:\n")
            f.write(f"  Mean: {np.mean(all_hrs):.2f} bpm\n")
            f.write(f"  Std: {np.std(all_hrs):.2f} bpm\n")
            f.write(f"  Min: {np.min(all_hrs):.2f} bpm\n")
            f.write(f"  Max: {np.max(all_hrs):.2f} bpm\n\n")
            
            f.write("Video Duration Statistics:\n")
            f.write(f"  Mean: {np.mean(all_durations):.2f} seconds\n")
            f.write(f"  Total: {np.sum(all_durations):.2f} seconds ({np.sum(all_durations)/60:.2f} minutes)\n\n")
            
            f.write("Subject Details:\n")
            f.write("-"*60 + "\n")
            for subject_name, data in sorted(annotations.items()):
                f.write(f"\n{subject_name}:\n")
                f.write(f"  Heart Rate: {data['heart_rate']:.2f} bpm\n")
                f.write(f"  Duration: {data['video_info']['duration']:.2f} seconds\n")
                f.write(f"  FPS: {data['video_info']['fps']:.2f}\n")
                f.write(f"  Resolution: {data['video_info']['width']}x{data['video_info']['height']}\n")
        
        logger.info(f"Dataset summary saved to: {output_file}")
    
    def visualize_dataset_statistics(self, annotations: Dict):
        """Create visualization of dataset statistics"""
        subjects = list(annotations.keys())
        heart_rates = [annotations[s]['heart_rate'] for s in subjects]
        durations = [annotations[s]['video_info']['duration'] for s in subjects]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Heart rate distribution
        axes[0, 0].hist(heart_rates, bins=20, edgecolor='black')
        axes[0, 0].set_title('Heart Rate Distribution')
        axes[0, 0].set_xlabel('Heart Rate (bpm)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].axvline(np.mean(heart_rates), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(heart_rates):.1f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Heart rate by subject
        axes[0, 1].bar(range(len(subjects)), heart_rates)
        axes[0, 1].set_title('Heart Rate by Subject')
        axes[0, 1].set_xlabel('Subject Index')
        axes[0, 1].set_ylabel('Heart Rate (bpm)')
        axes[0, 1].axhline(np.mean(heart_rates), color='r', linestyle='--')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Duration distribution
        axes[1, 0].hist(durations, bins=20, edgecolor='black')
        axes[1, 0].set_title('Video Duration Distribution')
        axes[1, 0].set_xlabel('Duration (seconds)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # HR variability
        hr_stds = [np.std(annotations[s]['hr_values']) for s in subjects]
        axes[1, 1].bar(range(len(subjects)), hr_stds)
        axes[1, 1].set_title('Heart Rate Variability by Subject')
        axes[1, 1].set_xlabel('Subject Index')
        axes[1, 1].set_ylabel('HR Std Dev (bpm)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset_statistics.png', dpi=150)
        plt.show()
        
        logger.info("Dataset statistics plot saved to: dataset_statistics.png")


def main():
    """Main function demonstrating usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process UBFC-rPPG Dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset folder (e.g., DATASET_2/)')
    parser.add_argument('--dataset_type', type=str, default='DATASET_2',
                       choices=['DATASET_1', 'DATASET_2'],
                       help='Dataset type')
    parser.add_argument('--output_json', type=str, default='annotations.json',
                       help='Output annotations file')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualizations for each subject')
    parser.add_argument('--stats', action='store_true',
                       help='Create statistics visualization')
    
    args = parser.parse_args()
    
    # Create processor
    processor = UBFCDatasetProcessor(args.dataset_path, args.dataset_type)
    
    # Process all subjects
    annotations = processor.process_all_subjects(
        visualize=args.visualize,
        output_json=args.output_json
    )
    
    if not annotations:
        logger.error("No annotations created. Check dataset structure.")
        return
    
    # Create summary
    processor.create_dataset_summary(annotations)
    
    # Create statistics visualization
    if args.stats:
        processor.visualize_dataset_statistics(annotations)
    
    logger.info("\n" + "="*60)
    logger.info("Dataset processing complete!")
    logger.info(f"Annotations saved to: {args.output_json}")
    logger.info("You can now use this with the training scripts:")
    logger.info(f"  python train_rppg.py --data_dir {args.dataset_path} --epochs 20")
    logger.info("="*60)


if __name__ == '__main__':
    main()