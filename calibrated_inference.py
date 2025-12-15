"""
Fixed Calibrated rPPG - Handles edge cases
"""

import cv2
import torch
import numpy as np
from collections import deque
import json
from pathlib import Path

from rppg_system import PhysNet, Config
from improved_hr_estimator import ImprovedHREstimator

class CalibratedRPPG:
    """Real-time rPPG with robust calibration"""
    
    def __init__(self, model_path, calibration_file='calibration.json'):
        self.device = torch.device('cpu')
        self.calibration_file = calibration_file
        
        # Load model
        self.model = PhysNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # HR estimator
        self.hr_estimator = ImprovedHREstimator(window_size=7)
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=Config.FRAMES_PER_WINDOW)
        
        # Load or initialize calibration
        self.calibration = self.load_calibration()
        
        print(f"✅ Model loaded")
        if self.calibration['calibrated']:
            print(f"📊 Calibration: {self.calibration['scale']:.3f}x + {self.calibration['offset']:.1f}")
        else:
            print(f"⚠️  Not calibrated - run calibration mode first")
    
    def load_calibration(self):
        """Load calibration parameters"""
        if Path(self.calibration_file).exists():
            with open(self.calibration_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'calibrated': False,
                'scale': 1.0,
                'offset': 0.0,
                'measurements': []
            }
    
    def save_calibration(self):
        """Save calibration parameters"""
        with open(self.calibration_file, 'w') as f:
            json.dump(self.calibration, f, indent=2)
        print(f"💾 Calibration saved to {self.calibration_file}")
    
    def calibrate(self):
        """Calibration mode - collect measurements"""
        print("\n" + "="*60)
        print("🔧 CALIBRATION MODE")
        print("="*60)
        print("\nInstructions:")
        print("1. Measure your actual HR with:")
        print("   - Pulse oximeter (most accurate)")
        print("   - Smartwatch/Fitbit")
        print("   - Manual (count pulse for 30 sec × 2)")
        print("\n2. We'll measure your HR 3 times")
        print("3. Make sure to change position/activity slightly between measurements")
        print("   (e.g., sit still, lean forward, lean back)")
        print("\n4. This helps get a better calibration")
        print("\nPress ENTER to start...")
        input()
        
        measurements = []
        
        for i in range(3):
            print(f"\n{'='*60}")
            print(f"MEASUREMENT {i+1}/3")
            print('='*60)
            
            if i == 0:
                print("Position: Sit normally, relaxed")
            elif i == 1:
                print("Position: Lean slightly forward or adjust position")
            else:
                print("Position: Try slightly different angle")
            
            print("\nPress ENTER when ready...")
            input()
            
            # Get predicted HR
            predicted_hr = self._collect_measurement()
            
            if predicted_hr is None:
                print("❌ Measurement failed, try again")
                i -= 1  # Retry this measurement
                continue
            
            # Get actual HR from user
            print(f"\n📊 System measured: {predicted_hr:.1f} bpm")
            
            while True:
                try:
                    actual_hr_input = input("\nEnter your ACTUAL HR (bpm): ")
                    actual_hr = float(actual_hr_input)
                    
                    if 40 <= actual_hr <= 200:
                        break
                    print("⚠️  Please enter a valid HR (40-200 bpm)")
                except ValueError:
                    print("⚠️  Please enter a number")
            
            error = actual_hr - predicted_hr
            measurements.append({
                'predicted': predicted_hr,
                'actual': actual_hr,
                'error': error
            })
            
            print(f"✅ Recorded: System={predicted_hr:.1f}, Actual={actual_hr:.1f}, Error={error:+.1f}")
        
        if len(measurements) >= 2:
            # Calculate calibration with robust method
            predicted = np.array([m['predicted'] for m in measurements])
            actual = np.array([m['actual'] for m in measurements])
            
            # Check variance
            pred_std = np.std(predicted)
            
            if pred_std < 2:  # Very low variance
                print("\n⚠️  Predicted values are very similar")
                print("This usually means you were very still in all measurements")
                print("\nUsing simple offset calibration...")
                
                # Use simple mean offset
                scale = 1.0
                offset = np.mean(actual - predicted)
                
            else:
                # Use linear regression
                # Method: minimize least squares
                # actual = scale * predicted + offset
                
                pred_mean = predicted.mean()
                act_mean = actual.mean()
                
                numerator = np.sum((predicted - pred_mean) * (actual - act_mean))
                denominator = np.sum((predicted - pred_mean) ** 2)
                
                if abs(denominator) < 1e-10:  # Still too small
                    scale = 1.0
                    offset = act_mean - pred_mean
                else:
                    scale = numerator / denominator
                    offset = act_mean - scale * pred_mean
            
            # Validate calibration
            if not (0.5 <= scale <= 2.0):
                print(f"\n⚠️  Unusual scale factor: {scale:.3f}")
                print("Using safer default calibration")
                scale = 1.0
                offset = np.mean(actual - predicted)
            
            self.calibration = {
                'calibrated': True,
                'scale': float(scale),
                'offset': float(offset),
                'measurements': measurements
            }
            
            self.save_calibration()
            
            print("\n" + "="*60)
            print("✅ CALIBRATION COMPLETE!")
            print("="*60)
            print(f"\nCalibration formula:")
            print(f"  Actual HR = {scale:.3f} × Measured + {offset:.1f}")
            
            # Show accuracy
            print(f"\nCalibration results:")
            errors_after = []
            for i, m in enumerate(measurements):
                calibrated = scale * m['predicted'] + offset
                error_before = m['actual'] - m['predicted']
                error_after = m['actual'] - calibrated
                errors_after.append(abs(error_after))
                
                print(f"\n  Measurement {i+1}:")
                print(f"    Actual HR:     {m['actual']:.1f} bpm")
                print(f"    Measured:      {m['predicted']:.1f} bpm (error: {error_before:+.1f})")
                print(f"    Calibrated:    {calibrated:.1f} bpm (error: {error_after:+.1f})")
            
            mean_error = np.mean(errors_after)
            print(f"\n  Average error after calibration: {mean_error:.1f} bpm")
            
            if mean_error < 5:
                print(f"  ✅ Excellent calibration!")
            elif mean_error < 10:
                print(f"  ✓ Good calibration")
            else:
                print(f"  ⚠️  Consider recalibrating with more varied measurements")
                
        else:
            print("\n❌ Need at least 2 valid measurements")
    
    def _collect_measurement(self):
        """Collect one HR measurement"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return None
        
        hr_samples = []
        stable_readings = 0
        last_hr = None
        frames_processed = 0
        max_frames = 1000  # Timeout after ~33 seconds
        
        print("\n📹 Measuring... sit still and face camera")
        print("Need 10 stable readings...")
        
        while stable_readings < 10 and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            display_frame = frame.copy()
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                margin = int(0.2 * w)
                x1, y1 = max(0, x-margin), max(0, y-margin)
                x2, y2 = min(frame.shape[1], x+w+margin), min(frame.shape[0], y+h+margin)
                
                face_roi = frame[y1:y2, x1:x2]
                
                # Process
                processed = self.process_frame(face_roi)
                self.frame_buffer.append(processed)
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if len(self.frame_buffer) == Config.FRAMES_PER_WINDOW:
                    frames = np.array(list(self.frame_buffer))
                    frames = torch.FloatTensor(frames).unsqueeze(0).transpose(1, 2)
                    frames = frames.to(self.device)
                    
                    with torch.no_grad():
                        rppg_pred = self.model(frames).cpu().numpy().squeeze()
                    
                    hr, _, _ = self.hr_estimator.estimate_hr(
                        rppg_pred, 
                        Config.FRAME_RATE, 
                        hr_range=(50, 120)
                    )
                    
                    # Check stability
                    if last_hr is not None:
                        diff = abs(hr - last_hr)
                        if diff < 3:  # Stable
                            stable_readings += 1
                            hr_samples.append(hr)
                            print(f"  ✓ Stable {stable_readings}/10: {hr:.1f} bpm", end='\r')
                        elif diff > 10:  # Very unstable, reset
                            stable_readings = 0
                            hr_samples = []
                        else:  # Somewhat unstable
                            stable_readings = max(0, stable_readings - 1)
                    
                    last_hr = hr
                    
                    cv2.putText(display_frame, f"HR: {hr:.1f} bpm", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(display_frame, f"Stable: {stable_readings}/10", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    progress = len(self.frame_buffer)
                    cv2.putText(display_frame, f"Warming up: {progress}/{Config.FRAMES_PER_WINDOW}", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No face detected", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Calibration - Measuring', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyAllWindows()
        
        print()  # New line after progress
        
        if len(hr_samples) >= 5:
            # Use median for robustness
            measured_hr = np.median(hr_samples)
            print(f"✅ Measurement complete: {measured_hr:.1f} bpm (from {len(hr_samples)} samples)")
            return measured_hr
        else:
            print(f"❌ Could not get stable reading (only {len(hr_samples)} stable samples)")
            return None
    
    def process_frame(self, frame):
        """Preprocess frame"""
        frame = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        return frame
    
    def apply_calibration(self, raw_hr):
        """Apply calibration to raw HR"""
        if self.calibration['calibrated']:
            calibrated = self.calibration['scale'] * raw_hr + self.calibration['offset']
            # Clamp to reasonable range
            return np.clip(calibrated, 40, 200)
        return raw_hr
    
    def run(self):
        """Run real-time detection with calibration"""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("🫀 Calibrated Heart Rate Detection")
        print("="*60)
        
        if not self.calibration['calibrated']:
            print("\n⚠️  WARNING: System not calibrated!")
            print("For accurate readings, run:")
            print("  python calibrated_inference.py <model> --calibrate")
            print("\nContinuing with raw measurements...\n")
        else:
            print(f"\n✅ Using calibration: {self.calibration['scale']:.3f}x + {self.calibration['offset']:.1f}")
        
        print("\n📋 Controls:")
        print("  Q: Quit")
        print("  R: Reset estimator")
        print("  C: Show calibration info")
        
        import time
        time.sleep(2)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                margin = int(0.2 * w)
                x1, y1 = max(0, x-margin), max(0, y-margin)
                x2, y2 = min(frame.shape[1], x+w+margin), min(frame.shape[0], y+h+margin)
                
                face_roi = frame[y1:y2, x1:x2]
                processed = self.process_frame(face_roi)
                self.frame_buffer.append(processed)
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if len(self.frame_buffer) == Config.FRAMES_PER_WINDOW:
                    frames = np.array(list(self.frame_buffer))
                    frames = torch.FloatTensor(frames).unsqueeze(0).transpose(1, 2)
                    frames = frames.to(self.device)
                    
                    with torch.no_grad():
                        rppg_pred = self.model(frames).cpu().numpy().squeeze()
                    
                    raw_hr, _, _ = self.hr_estimator.estimate_hr(
                        rppg_pred, 
                        Config.FRAME_RATE, 
                        hr_range=(50, 120)
                    )
                    
                    # Apply calibration
                    calibrated_hr = self.apply_calibration(raw_hr)
                    
                    # Display
                    if self.calibration['calibrated']:
                        cv2.putText(display_frame, f"HR: {calibrated_hr:.1f} bpm", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1.5, (0, 255, 0), 3)
                        cv2.putText(display_frame, f"Raw: {raw_hr:.1f}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (180, 180, 180), 2)
                        cv2.putText(display_frame, "✓ CALIBRATED", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, f"HR: {raw_hr:.1f} bpm", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1.5, (0, 165, 255), 3)
                        cv2.putText(display_frame, "⚠ NOT CALIBRATED", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 165, 255), 2)
                else:
                    progress = len(self.frame_buffer)
                    cv2.putText(display_frame, f"Starting: {progress}/{Config.FRAMES_PER_WINDOW}", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "No face detected", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2)
            
            cv2.imshow('Calibrated rPPG', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.hr_estimator.reset()
                self.frame_buffer.clear()
                print("🔄 Reset")
            elif key == ord('c'):
                if self.calibration['calibrated']:
                    print(f"\n📊 Calibration Info:")
                    print(f"  Formula: {self.calibration['scale']:.3f}x + {self.calibration['offset']:.1f}")
                    print(f"  Measurements: {len(self.calibration['measurements'])}")
                else:
                    print("\n⚠️  Not calibrated - run with --calibrate flag")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrated rPPG Heart Rate Detection')
    parser.add_argument('model_path', help='Path to trained model (.pth file)')
    parser.add_argument('--calibrate', action='store_true', 
                       help='Run calibration mode (do this first!)')
    args = parser.parse_args()
    
    app = CalibratedRPPG(args.model_path)
    
    if args.calibrate:
        app.calibrate()
        print("\n✅ Calibration saved!")
        print("\nNow run without --calibrate:")
        print(f"  python calibrated_inference.py {args.model_path}")
    else:
        app.run()