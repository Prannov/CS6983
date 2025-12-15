"""
rPPG Heart Rate Detection - GUI Application
Simple interface for real-time and video processing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import torch
import numpy as np
from collections import deque
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from pathlib import Path
import json

from rppg_system import PhysNet, Config
from improved_hr_estimator import ImprovedHREstimator

class RPPGApp:
    """Main GUI Application"""
    
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("rPPG Heart Rate Detection")
        self.root.geometry("1400x850")
        
        # Variables
        self.model_path = model_path
        self.is_running = False
        self.mode = "realtime"
        self.video_path = None
        
        # HR data
        self.hr_history = deque(maxlen=500)  # Store more data
        self.time_history = deque(maxlen=500)
        self.start_time = None
        
        # Initialize model
        self.init_model()
        
        # Create UI
        self.create_ui()
        
        # Load calibration
        self.load_calibration()
    
    def init_model(self):
        """Initialize model and components"""
        self.device = torch.device('cpu')
        
        # Load model
        self.model = PhysNet()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        # Face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # HR estimator
        self.hr_estimator = ImprovedHREstimator(window_size=7)
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=Config.FRAMES_PER_WINDOW)
        
        # Calibration
        self.calibration = {'calibrated': False, 'scale': 1.0, 'offset': 0.0}
    
    def create_ui(self):
        """Create the user interface"""
        
        # ============================================================
        # Top Frame - Controls
        # ============================================================
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Label(top_frame, text="rPPG Heart Rate Detection", 
                 font=('Helvetica', 18, 'bold')).grid(row=0, column=0, columnspan=4, pady=10)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(top_frame, text="Mode", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        self.mode_var = tk.StringVar(value="realtime")
        ttk.Radiobutton(mode_frame, text="Real-time (Webcam)", 
                       variable=self.mode_var, value="realtime",
                       command=self.on_mode_change).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(mode_frame, text="Video File", 
                       variable=self.mode_var, value="video",
                       command=self.on_mode_change).grid(row=0, column=1, padx=10)
        
        # Video file selection
        self.video_frame = ttk.Frame(mode_frame)
        self.video_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.video_label = ttk.Label(self.video_frame, text="No video selected")
        self.video_label.grid(row=0, column=0, padx=5)
        
        ttk.Button(self.video_frame, text="Browse", 
                  command=self.browse_video).grid(row=0, column=1, padx=5)
        
        self.video_frame.grid_remove()
        
        # Control buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=15)
        
        self.start_button = ttk.Button(button_frame, text="Start", 
                                       command=self.start_detection,
                                       width=12)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                      command=self.stop_detection,
                                      state='disabled',
                                      width=12)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="Save Results", 
                  command=self.save_results,
                  width=12).grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text="Calibrate", 
                  command=self.open_calibration,
                  width=12).grid(row=0, column=3, padx=5)
        
        ttk.Button(button_frame, text="Help", 
                  command=self.show_help,
                  width=12).grid(row=0, column=4, padx=5)
        
        # ============================================================
        # Middle Left - Video Display
        # ============================================================
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video canvas
        self.video_canvas = tk.Canvas(left_frame, width=640, height=480, bg='black')
        self.video_canvas.pack()
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Select mode and click Start")
        self.status_label = ttk.Label(left_frame, textvariable=self.status_var,
                                      font=('Helvetica', 10))
        self.status_label.pack(pady=5)
        
        # HR Display - Large
        hr_display_frame = ttk.LabelFrame(left_frame, text="Current Heart Rate", padding="15")
        hr_display_frame.pack(pady=10, fill=tk.X)
        
        self.hr_var = tk.StringVar(value="-- bpm")
        hr_display = ttk.Label(hr_display_frame, textvariable=self.hr_var,
                               font=('Helvetica', 48, 'bold'),
                               foreground='green')
        hr_display.pack()
        
        # Calibration status
        self.calib_var = tk.StringVar(value="Not Calibrated")
        calib_label = ttk.Label(hr_display_frame, textvariable=self.calib_var,
                               font=('Helvetica', 10),
                               foreground='orange')
        calib_label.pack(pady=5)
        
        # ============================================================
        # Middle Right - Statistics and Graph
        # ============================================================
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Statistics display
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=40, font=('Courier', 10))
        self.stats_text.pack()
        self.stats_text.insert('1.0', 'No data yet...')
        self.stats_text.config(state='disabled')
        
        # Graph
        graph_frame = ttk.LabelFrame(right_frame, text="Heart Rate Over Time", padding="10")
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot
        self.line, = self.ax.plot([], [], 'g-', linewidth=2, label='Heart Rate')
        self.ax.set_xlabel('Time (seconds)', fontsize=10)
        self.ax.set_ylabel('Heart Rate (bpm)', fontsize=10)
        self.ax.set_title('Heart Rate Monitoring', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, 60)  # Start with 60 seconds view
        self.ax.set_ylim(40, 140)
        self.ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)
    
    def load_calibration(self):
        """Load calibration from file"""
        calib_file = Path('calibration.json')
        if calib_file.exists():
            try:
                with open(calib_file, 'r') as f:
                    self.calibration = json.load(f)
                
                if self.calibration.get('calibrated'):
                    self.calib_var.set(f"Calibrated ({self.calibration['scale']:.2f}x + {self.calibration['offset']:.1f})")
            except Exception as e:
                print(f"Error loading calibration: {e}")
    
    def on_mode_change(self):
        """Handle mode change"""
        mode = self.mode_var.get()
        if mode == "video":
            self.video_frame.grid()
        else:
            self.video_frame.grid_remove()
    
    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.video_path = filename
            self.video_label.config(text=f"{Path(filename).name}")
    
    def start_detection(self):
        """Start heart rate detection"""
        mode = self.mode_var.get()
        
        if mode == "video" and not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file first")
            return
        
        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Clear data
        self.hr_history.clear()
        self.time_history.clear()
        self.start_time = time.time()
        self.frame_buffer.clear()
        self.hr_estimator.reset()
        
        # Reset graph
        self.line.set_data([], [])
        self.ax.set_xlim(0, 60)
        self.canvas.draw()
        
        # Start processing thread
        if mode == "realtime":
            thread = threading.Thread(target=self.process_realtime, daemon=True)
        else:
            thread = threading.Thread(target=self.process_video, daemon=True)
        
        thread.start()
    
    def stop_detection(self):
        """Stop heart rate detection"""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_var.set("Stopped")
    
    def process_realtime(self):
        """Process real-time webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera")
            self.stop_detection()
            return
        
        self.status_var.set("Live - Real-time Detection")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            hr = self.process_frame(frame)
            
            # Display
            self.display_frame(frame)
            
            if hr is not None:
                self.update_hr_display(hr)
            
            time.sleep(0.01)
        
        cap.release()
        self.status_var.set("Stopped")
    
    def process_video(self):
        """Process video file"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video")
            self.stop_detection()
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            hr = self.process_frame(frame)
            
            # Display
            self.display_frame(frame)
            
            if hr is not None:
                self.update_hr_display(hr)
            
            # Update progress
            progress = (frame_count / total_frames) * 100
            self.status_var.set(f"Processing video... {progress:.1f}%")
            
            time.sleep(1.0 / fps if fps > 0 else 0.03)
        
        cap.release()
        self.status_var.set("Video processing complete")
        self.stop_detection()
    
    def process_frame(self, frame):
        """Process a single frame and return HR"""
        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        
        # Extract face ROI
        margin = int(0.2 * w)
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(frame.shape[1], x+w+margin), min(frame.shape[0], y+h+margin)
        
        face_roi = frame[y1:y2, x1:x2]
        
        # Draw rectangle on original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add "Face Detected" indicator
        cv2.putText(frame, "Face Detected", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Process face
        face_processed = cv2.resize(face_roi, (Config.IMG_SIZE, Config.IMG_SIZE))
        face_processed = face_processed.astype(np.float32) / 255.0
        face_processed = np.transpose(face_processed, (2, 0, 1))
        
        self.frame_buffer.append(face_processed)
        
        # Show buffer progress on frame
        buffer_progress = len(self.frame_buffer)
        if buffer_progress < Config.FRAMES_PER_WINDOW:
            cv2.putText(frame, f"Warming up: {buffer_progress}/{Config.FRAMES_PER_WINDOW}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Estimate HR when buffer is full
        if len(self.frame_buffer) == Config.FRAMES_PER_WINDOW:
            frames = np.array(list(self.frame_buffer))
            frames = torch.FloatTensor(frames).unsqueeze(0).transpose(1, 2)
            frames = frames.to(self.device)
            
            with torch.no_grad():
                rppg_pred = self.model(frames).cpu().numpy().squeeze()
            
            raw_hr, _, _ = self.hr_estimator.estimate_hr(rppg_pred, Config.FRAME_RATE, hr_range=(50, 120))
            
            # Apply calibration
            if self.calibration.get('calibrated'):
                hr = self.calibration['scale'] * raw_hr + self.calibration['offset']
                hr = np.clip(hr, 40, 200)
            else:
                hr = raw_hr
            
            return hr
        
        return None
    
    def display_frame(self, frame):
        """Display frame on canvas"""
        # Resize frame to fit canvas
        frame = cv2.resize(frame, (640, 480))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.video_canvas.image = img_tk
    
    def update_hr_display(self, hr):
        """Update HR display and graph"""
        # Update HR text
        self.hr_var.set(f"{hr:.1f} bpm")
        
        # Add to history
        current_time = time.time() - self.start_time
        self.hr_history.append(hr)
        self.time_history.append(current_time)
        
        # Update statistics
        self.update_statistics()
        
        # Update graph every 0.5 seconds
        if len(self.hr_history) % 15 == 0:  # Update less frequently for performance
            self.update_graph()
    
    def update_statistics(self):
        """Update statistics display"""
        if len(self.hr_history) == 0:
            return
        
        hrs = list(self.hr_history)
        
        stats = f"""
Duration:    {max(self.time_history):.1f} seconds
Samples:     {len(hrs)}

Current HR:  {hrs[-1]:.1f} bpm
Average HR:  {np.mean(hrs):.1f} bpm
Min HR:      {min(hrs):.1f} bpm
Max HR:      {max(hrs):.1f} bpm
Std Dev:     {np.std(hrs):.1f} bpm
        """
        
        self.stats_text.config(state='normal')
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats)
        self.stats_text.config(state='disabled')
    
    def update_graph(self):
        """Update the HR graph"""
        if len(self.time_history) < 2:
            return
        
        times = list(self.time_history)
        hrs = list(self.hr_history)
        
        # Update data
        self.line.set_data(times, hrs)
        
        # Auto-scale x-axis
        max_time = max(times)
        if max_time > self.ax.get_xlim()[1] - 10:
            # Expand x-axis
            self.ax.set_xlim(0, max_time + 20)
        
        # Auto-scale y-axis based on data
        if len(hrs) > 10:
            hr_min = max(40, min(hrs) - 10)
            hr_max = min(200, max(hrs) + 10)
            self.ax.set_ylim(hr_min, hr_max)
        
        # Redraw
        self.canvas.draw()
        self.canvas.flush_events()
    
    def save_results(self):
        """Save HR data and graph"""
        if len(self.hr_history) == 0:
            messagebox.showinfo("No Data", "No heart rate data to save")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ],
            title="Save Results As",
            initialfile="rppg_results"
        )
        
        if not filename:
            return
        
        base_path = Path(filename).with_suffix('')
        
        try:
            # Save graph
            graph_path = str(base_path) + '_graph.png'
            self.fig.savefig(graph_path, dpi=300, bbox_inches='tight')
            
            # Save data
            data_path = str(base_path) + '_data.txt'
            with open(data_path, 'w') as f:
                f.write("Time(s)\tHeart_Rate(bpm)\n")
                for t, hr in zip(self.time_history, self.hr_history):
                    f.write(f"{t:.2f}\t{hr:.2f}\n")
            
            # Save CSV for Excel
            csv_path = str(base_path) + '_data.csv'
            with open(csv_path, 'w') as f:
                f.write("Time_seconds,Heart_Rate_bpm\n")
                for t, hr in zip(self.time_history, self.hr_history):
                    f.write(f"{t:.2f},{hr:.2f}\n")
            
            # Save summary
            summary_path = str(base_path) + '_summary.txt'
            hrs = list(self.hr_history)
            
            with open(summary_path, 'w') as f:
                f.write("rPPG Heart Rate Detection - Summary Report\n")
                f.write("="*60 + "\n\n")
                f.write(f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mode: {self.mode_var.get()}\n")
                if self.mode_var.get() == 'video':
                    f.write(f"Video: {Path(self.video_path).name}\n")
                f.write("\n" + "-"*60 + "\n")
                f.write("STATISTICS\n")
                f.write("-"*60 + "\n")
                f.write(f"Duration:        {max(self.time_history):.1f} seconds\n")
                f.write(f"Total Samples:   {len(hrs)}\n")
                f.write(f"Average HR:      {np.mean(hrs):.1f} bpm\n")
                f.write(f"Minimum HR:      {min(hrs):.1f} bpm\n")
                f.write(f"Maximum HR:      {max(hrs):.1f} bpm\n")
                f.write(f"Std Deviation:   {np.std(hrs):.1f} bpm\n")
                f.write(f"Variance:        {np.var(hrs):.1f}\n")
                
                if self.calibration.get('calibrated'):
                    f.write(f"\n" + "-"*60 + "\n")
                    f.write("CALIBRATION\n")
                    f.write("-"*60 + "\n")
                    f.write(f"Calibrated:      Yes\n")
                    f.write(f"Scale Factor:    {self.calibration['scale']:.3f}\n")
                    f.write(f"Offset:          {self.calibration['offset']:.1f} bpm\n")
                    f.write(f"Formula:         HR = {self.calibration['scale']:.3f} × Raw + {self.calibration['offset']:.1f}\n")
                else:
                    f.write(f"\nCalibrated:      No (consider calibrating for better accuracy)\n")
            
            messagebox.showinfo("Saved Successfully", 
                               f"Results saved:\n\n"
                               f"Graph: {Path(graph_path).name}\n"
                               f"Data (TXT): {Path(data_path).name}\n"
                               f"Data (CSV): {Path(csv_path).name}\n"
                               f"Summary: {Path(summary_path).name}\n\n"
                               f"Location: {base_path.parent}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results:\n{e}")
    
    def open_calibration(self):
        """Open calibration dialog"""
        CalibrationDialog(self.root, self)
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
rPPG Heart Rate Detection - User Guide

GETTING STARTED:
1. Select mode: Real-time (webcam) or Video File
2. If Video mode, click Browse to select a video
3. Click Start to begin detection
4. Heart rate will update in real-time
5. Click Stop when finished
6. Click Save Results to export

CALIBRATION (Recommended):
- Click "Calibrate" button for accurate measurements
- You'll need a reference HR (smartwatch/oximeter)
- Follow the 3-step calibration wizard
- Only needs to be done once

REAL-TIME MODE:
- Uses your webcam
- Live heart rate monitoring
- Best for: Health monitoring, demos

VIDEO MODE:
- Process pre-recorded videos
- Useful for: Analysis, batch processing
- Supports: MP4, AVI, MOV files

TIPS FOR ACCURACY:
Good lighting (bright, diffuse)
Face camera directly
Stay still for 15-20 seconds
Calibrate the system first
No glasses/hats for best results

TROUBLESHOOTING:
- No face detected: Improve lighting, adjust position
- Unstable readings: Stay still, wait longer
- Inaccurate HR: Run calibration mode

SAVING RESULTS:
Files saved include:
- Graph image (high-res PNG)
- Data file (TXT - tab separated)
- Data file (CSV - for Excel)
- Summary report (TXT - statistics)

For support or questions, check the documentation.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help & User Guide")
        help_window.geometry("600x700")
        
        # Scrollable text
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                      font=('Helvetica', 10), padx=15, pady=15)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text.yview)
        
        text.insert('1.0', help_text)
        text.config(state='disabled')
        
        ttk.Button(help_window, text="Close", 
                  command=help_window.destroy).pack(pady=10)


class CalibrationDialog:
    """Simple calibration dialog"""
    
    def __init__(self, parent, app):
        self.app = app
        
        self.window = tk.Toplevel(parent)
        self.window.title("System Calibration")
        self.window.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Calibration Wizard", 
                 font=('Helvetica', 16, 'bold')).pack(pady=10)
        
        # Instructions
        instructions = """
To calibrate the system for accurate measurements:

1. You'll need a reference heart rate measurement
   (pulse oximeter, smartwatch, or manual count)

2. The system will measure your HR 3 times

3. Enter your actual HR for each measurement

4. System calculates correction factors

This ensures accurate readings for your specific:
- Lighting conditions
- Camera setup
- Skin tone
- Environment

Calibration takes about 3-4 minutes.

Ready to begin?
        """
        
        text = tk.Text(main_frame, wrap=tk.WORD, height=15, 
                      font=('Helvetica', 11), padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True, pady=10)
        text.insert('1.0', instructions)
        text.config(state='disabled')
        
        # Note
        note_frame = ttk.LabelFrame(main_frame, text="Important", padding="10")
        note_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(note_frame, text="Make sure you have a way to measure your actual heart rate!",
                 foreground='red', font=('Helvetica', 10, 'bold')).pack()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=15)
        
        ttk.Button(button_frame, text="Start Calibration", 
                  command=self.start_calibration,
                  width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=self.window.destroy,
                  width=20).pack(side=tk.LEFT, padx=5)
    
    def start_calibration(self):
        """Run calibration in terminal"""
        self.window.destroy()
        
        messagebox.showinfo("Calibration Started", 
                           "Calibration is running in your terminal/console.\n\n"
                           "Please follow the instructions there.\n\n"
                           "The GUI will update when calibration is complete.")
        
        import subprocess
        import sys
        
        # Run calibration script
        subprocess.run([sys.executable, 'calibrated_inference.py', 
                       self.app.model_path, '--calibrate'])
        
        # Reload calibration
        self.app.load_calibration()
        
        messagebox.showinfo("Calibration Complete", 
                           "Calibration successful!\n\n"
                           "The system is now calibrated for accurate measurements.")


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        # Try to find model automatically
        model_path = Path('./final_model.pth')
        if not model_path.exists():
            model_path = Path('./outputs/ubfc_model/final_model.pth')
        
        if not model_path.exists():
            print("Usage: python rppg_gui.py <model_path>")
            print("Example: python rppg_gui.py ./final_model.pth")
            sys.exit(1)
        
        model_path = str(model_path)
    else:
        model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print("="*60)
    print("rPPG Heart Rate Detection - GUI")
    print("="*60)
    print(f"\nModel: {model_path}")
    print("Starting GUI...")
    
    root = tk.Tk()
    app = RPPGApp(root, model_path)
    root.mainloop()


if __name__ == '__main__':
    main()