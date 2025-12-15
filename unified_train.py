"""
Unified Training Script for rPPG Heart Rate Detection
Supports: UBFC-rPPG, Google Drive, Local files

Usage Examples:
    # UBFC dataset (local)
    python unified_train.py --dataset ubfc --data_dir ./DATASET_2 --epochs 20
    
    # Google Drive (public)
    python unified_train.py --dataset gdrive_public \
        --folder_url "YOUR_URL" --annotations_url "YOUR_URL" --epochs 20
    
    # Google Drive (private, streaming)
    python unified_train.py --dataset gdrive_private \
        --folder_id "YOUR_ID" --annotations_id "YOUR_ID" --stream --epochs 20
    
    # Local dataset with custom structure
    python unified_train.py --dataset local --data_dir ./my_dataset --epochs 20
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Import our modules
from rppg_system import (
    Config, PhysNet, Trainer, HeartRateEstimator, 
    Metrics, rPPGDataset
)

try:
    from rppg_gdrive_loader import (
        GDriveFileHandler, GDriveRPPGDataset, SimpleGDriveDownloader
    )
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    logging.warning("Google Drive support not available")

try:
    from ubfc_dataset_processor import UBFCDatasetProcessor
    UBFC_AVAILABLE = True
except ImportError:
    UBFC_AVAILABLE = False
    logging.warning("UBFC processor not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDatasetLoader:
    """Unified loader for all dataset types"""
    
    @staticmethod
    def load_ubfc_dataset(data_dir: str, cache_annotations: bool = True):
        """Load UBFC-rPPG dataset"""
        if not UBFC_AVAILABLE:
            raise ImportError("UBFC processor not available. Ensure ubfc_dataset_processor.py is present.")
        
        logger.info("Loading UBFC-rPPG dataset...")
        
        annotations_file = Path(data_dir) / 'annotations.json'
        
        # Check if annotations already exist
        if annotations_file.exists() and cache_annotations:
            logger.info(f"Loading cached annotations from {annotations_file}")
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
        else:
            # Process dataset
            logger.info("Processing UBFC dataset (this may take a while)...")
            processor = UBFCDatasetProcessor(data_dir)
            annotations = processor.process_all_subjects(
                visualize=False,
                output_json=str(annotations_file)
            )
        
        # Prepare for dataset
        video_paths = []
        ppg_signals = []
        hr_labels = []
        
        for subject_name, data in annotations.items():
            video_paths.append(data['video_path'])
            ppg_signals.append(np.array(data['ppg']))
            hr_labels.append(data['heart_rate'])
        
        logger.info(f"Loaded {len(video_paths)} videos from UBFC dataset")
        
        return video_paths, ppg_signals, hr_labels, annotations
    
    @staticmethod
    def load_gdrive_public(folder_url: str, annotations_url: str, cache_dir: str = './cache'):
        """Load from public Google Drive"""
        if not GDRIVE_AVAILABLE:
            raise ImportError("Google Drive support not available")
        
        logger.info("Loading from public Google Drive...")
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Download annotations
        annotations_file = cache_path / 'annotations.json'
        SimpleGDriveDownloader.download_public_file(annotations_url, str(annotations_file))
        
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Download videos folder
        videos_dir = cache_path / 'videos'
        if not videos_dir.exists():
            logger.info("Downloading videos from Google Drive (this may take a while)...")
            SimpleGDriveDownloader.download_public_folder(folder_url, str(videos_dir))
        else:
            logger.info("Using cached videos")
        
        # Prepare dataset
        video_paths = []
        ppg_signals = []
        hr_labels = []
        
        for video_file in videos_dir.glob('*.mp4'):
            key = video_file.stem
            if key in annotations:
                video_paths.append(str(video_file))
                ppg_signals.append(np.array(annotations[key]['ppg']))
                hr_labels.append(annotations[key]['heart_rate'])
        
        logger.info(f"Loaded {len(video_paths)} videos from Google Drive")
        
        return video_paths, ppg_signals, hr_labels, annotations
    
    @staticmethod
    def load_gdrive_private(folder_id: str, annotations_id: str, use_streaming: bool = True):
        """Load from private Google Drive"""
        if not GDRIVE_AVAILABLE:
            raise ImportError("Google Drive support not available")
        
        logger.info("Loading from private Google Drive...")
        
        gdrive = GDriveFileHandler()
        
        # Load annotations
        annotations_data = gdrive.download_file_to_memory(annotations_id)
        annotations = json.loads(annotations_data.decode('utf-8'))
        
        # List videos
        files = gdrive.list_folder_contents(folder_id)
        video_files = [
            {'id': f['id'], 'name': f['name']}
            for f in files
            if f['name'].endswith(('.mp4', '.avi', '.mov'))
        ]
        
        if use_streaming:
            # Return dataset that streams from Drive
            logger.info("Using streaming mode (no download)")
            return None, None, None, {
                'gdrive': gdrive,
                'video_files': video_files,
                'annotations': annotations,
                'streaming': True
            }
        else:
            # Download videos
            cache_dir = Path('./cache/videos')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            video_paths = []
            ppg_signals = []
            hr_labels = []
            
            for video_info in video_files:
                key = Path(video_info['name']).stem
                if key in annotations:
                    # Download if not cached
                    video_path = cache_dir / video_info['name']
                    if not video_path.exists():
                        gdrive.download_file_to_disk(video_info['id'], str(video_path))
                    
                    video_paths.append(str(video_path))
                    ppg_signals.append(np.array(annotations[key]['ppg']))
                    hr_labels.append(annotations[key]['heart_rate'])
            
            return video_paths, ppg_signals, hr_labels, annotations
    
    @staticmethod
    def load_local_dataset(data_dir: str, annotations_file: str = 'annotations.json'):
        """Load from local directory"""
        logger.info(f"Loading local dataset from {data_dir}")
        
        data_path = Path(data_dir)
        annotations_path = data_path / annotations_file
        
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Find videos
        video_paths = []
        ppg_signals = []
        hr_labels = []
        
        for video_file in data_path.glob('**/*.mp4'):
            key = video_file.stem
            if key in annotations:
                video_paths.append(str(video_file))
                ppg_signals.append(np.array(annotations[key]['ppg']))
                hr_labels.append(annotations[key]['heart_rate'])
        
        logger.info(f"Loaded {len(video_paths)} videos from local dataset")
        
        return video_paths, ppg_signals, hr_labels, annotations


def create_dataloaders(video_paths, ppg_signals, hr_labels, 
                       batch_size=8, train_split=0.8, 
                       gdrive_info=None):
    """Create train and validation dataloaders"""
    
    if gdrive_info and gdrive_info.get('streaming'):
        # Use streaming dataset
        logger.info("Creating streaming datasets...")
        
        # Split files
        video_files = gdrive_info['video_files']
        n_train = int(len(video_files) * train_split)
        np.random.shuffle(video_files)
        
        train_dataset = GDriveRPPGDataset(
            folder_id_or_file_list=video_files[:n_train],
            annotations=gdrive_info['annotations'],
            gdrive_handler=gdrive_info['gdrive']
        )
        
        val_dataset = GDriveRPPGDataset(
            folder_id_or_file_list=video_files[n_train:],
            annotations=gdrive_info['annotations'],
            gdrive_handler=gdrive_info['gdrive']
        )
    else:
        # Use standard dataset
        logger.info("Creating standard datasets...")
        
        dataset = rPPGDataset(video_paths, ppg_signals, hr_labels)
        
        # Split
        n_train = int(len(dataset) * train_split)
        n_val = len(dataset) - n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for streaming
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def evaluate_final_model(model, val_loader, device=Config.DEVICE):
    """Comprehensive evaluation of trained model"""
    logger.info("Evaluating final model...")
    
    model.eval()
    hr_estimator = HeartRateEstimator()
    metrics = Metrics()
    
    all_hr_pred = []
    all_hr_true = []
    all_rppg_signals = []
    
    with torch.no_grad():
        for batch in val_loader:
            frames = batch['frames'].to(device)
            ppg_target = batch['ppg'].cpu().numpy()
            hr_target = batch['hr'].cpu().numpy()
            
            # Predict
            rppg_pred = model(frames).cpu().numpy()
            
            for i in range(len(rppg_pred)):
                rppg = rppg_pred[i]
                hr_true = hr_target[i][0]
                
                # Estimate HR
                hr_est, _, _ = hr_estimator.estimate_hr(rppg, Config.FRAME_RATE)
                
                all_hr_pred.append(hr_est)
                all_hr_true.append(hr_true)
                all_rppg_signals.append(rppg)
    
    # Convert to arrays
    all_hr_pred = np.array(all_hr_pred)
    all_hr_true = np.array(all_hr_true)
    
    # Calculate metrics
    mae = metrics.mae(all_hr_pred, all_hr_true)
    r = metrics.pearson_correlation(all_hr_pred, all_hr_true)
    
    # SNR and TMC
    snr_values = []
    tmc_values = []
    for rppg, hr_true in zip(all_rppg_signals, all_hr_true):
        snr = metrics.snr(rppg, hr_true, Config.FRAME_RATE)
        tmc = metrics.tmc(rppg, Config.FRAME_RATE)
        snr_values.append(snr)
        tmc_values.append(tmc)
    
    mean_snr = np.mean(snr_values)
    mean_tmc = np.mean(tmc_values)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"MAE:       {mae:.2f} bpm")
    logger.info(f"Pearson r: {r:.3f}")
    logger.info(f"SNR:       {mean_snr:.2f} dB")
    logger.info(f"TMC:       {mean_tmc:.3f}")
    logger.info("="*60)
    
    return {
        'MAE': mae,
        'Pearson_r': r,
        'SNR': mean_snr,
        'TMC': mean_tmc,
        'hr_pred': all_hr_pred,
        'hr_true': all_hr_true
    }


def plot_results(train_losses, val_losses, eval_results, output_dir):
    """Create result visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Curves', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # HR predictions
    hr_true = eval_results['hr_true']
    hr_pred = eval_results['hr_pred']
    
    axes[1].scatter(hr_true, hr_pred, alpha=0.5)
    axes[1].plot([40, 240], [40, 240], 'r--', linewidth=2, label='Perfect')
    axes[1].set_xlabel('True HR (bpm)', fontsize=12)
    axes[1].set_ylabel('Predicted HR (bpm)', fontsize=12)
    axes[1].set_title(f'HR Predictions (MAE: {eval_results["MAE"]:.2f} bpm)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_results.png', dpi=150, bbox_inches='tight')
    logger.info(f"Results plot saved to {output_path / 'training_results.png'}")
    plt.close()


def main(args):
    """Main training function"""
    
    logger.info("="*60)
    logger.info("Unified rPPG Training Pipeline")
    logger.info("="*60)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset based on type
    gdrive_info = None
    
    if args.dataset == 'ubfc':
        video_paths, ppg_signals, hr_labels, annotations = \
            UnifiedDatasetLoader.load_ubfc_dataset(args.data_dir)
    
    elif args.dataset == 'gdrive_public':
        video_paths, ppg_signals, hr_labels, annotations = \
            UnifiedDatasetLoader.load_gdrive_public(
                args.folder_url, args.annotations_url, args.cache_dir
            )
    
    elif args.dataset == 'gdrive_private':
        video_paths, ppg_signals, hr_labels, gdrive_info = \
            UnifiedDatasetLoader.load_gdrive_private(
                args.folder_id, args.annotations_id, args.stream
            )
        if gdrive_info.get('streaming'):
            annotations = gdrive_info['annotations']
    
    elif args.dataset == 'local':
        video_paths, ppg_signals, hr_labels, annotations = \
            UnifiedDatasetLoader.load_local_dataset(args.data_dir)
    
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        video_paths, ppg_signals, hr_labels,
        batch_size=args.batch_size,
        train_split=args.train_split,
        gdrive_info=gdrive_info
    )
    
    # Create model
    logger.info("Creating model...")
    model = PhysNet(in_channels=3 if args.color else 1)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: PhysNet with {total_params:,} parameters")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device=Config.DEVICE)
    train_losses, val_losses = trainer.train(num_epochs=args.epochs)
    
    # Final evaluation
    eval_results = evaluate_final_model(model, val_loader)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    logger.info(f"\nModel saved to {output_dir / 'final_model.pth'}")
    
    # Save metrics
    results = {
        'dataset': args.dataset,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_metrics': {
            'MAE': float(eval_results['MAE']),
            'Pearson_r': float(eval_results['Pearson_r']),
            'SNR': float(eval_results['SNR']),
            'TMC': float(eval_results['TMC'])
        },
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_split': args.train_split
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(train_losses, val_losses, eval_results, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified rPPG Training')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ubfc', 'gdrive_public', 'gdrive_private', 'local'],
                       help='Dataset type')
    
    # Common arguments
    parser.add_argument('--data_dir', type=str,
                       help='Data directory (for ubfc/local)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Cache directory')
    
    # Google Drive public
    parser.add_argument('--folder_url', type=str,
                       help='Google Drive folder URL (public)')
    parser.add_argument('--annotations_url', type=str,
                       help='Annotations file URL (public)')
    
    # Google Drive private
    parser.add_argument('--folder_id', type=str,
                       help='Google Drive folder ID (private)')
    parser.add_argument('--annotations_id', type=str,
                       help='Annotations file ID (private)')
    parser.add_argument('--stream', action='store_true',
                       help='Stream from Google Drive (private only)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/validation split')
    
    # Model arguments
    parser.add_argument('--color', action='store_true',
                       help='Use RGB (default: grayscale for NIR)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset == 'ubfc' and not args.data_dir:
        parser.error("--dataset ubfc requires --data_dir")
    if args.dataset == 'gdrive_public' and (not args.folder_url or not args.annotations_url):
        parser.error("--dataset gdrive_public requires --folder_url and --annotations_url")
    if args.dataset == 'gdrive_private' and (not args.folder_id or not args.annotations_id):
        parser.error("--dataset gdrive_private requires --folder_id and --annotations_id")
    if args.dataset == 'local' and not args.data_dir:
        parser.error("--dataset local requires --data_dir")
    
    # Update config
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    
    main(args)