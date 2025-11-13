"""
Real-time Training Monitor
Reads train_model.py logs and displays live training metrics in graphs

Usage:
    python retrain/monitoring_train.py
    python retrain/monitoring_train.py --log-file retrain/logs/train_model_20251112.log
    python retrain/monitoring_train.py --fresh  # Start from current position (skip existing data)
"""

import re
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# Set up matplotlib for real-time plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.ion()  # Enable interactive mode

class TrainingMonitor:
    def __init__(self, log_file=None, start_fresh=False):
        self.base_path = Path(__file__).parent.parent
        self.log_dir = self.base_path / "retrain" / "logs"
        self.start_fresh = start_fresh

        # Auto-detect latest log file if not specified
        if log_file is None:
            log_file = self._find_latest_log()

        self.log_file = Path(log_file) if log_file else None

        if not self.log_file or not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            print(f"   Expected location: {self.log_dir}")
            sys.exit(1)

        print(f"📊 Monitoring log file: {self.log_file}")
        if start_fresh:
            print(f"   Mode: Fresh start (skipping existing data)")

        # Data storage
        self.epochs = []
        self.train_box_loss = []
        self.train_cls_loss = []
        self.train_dfl_loss = []
        self.val_box_loss = []
        self.val_cls_loss = []
        self.val_dfl_loss = []
        self.precision = []
        self.recall = []
        self.map50 = []
        self.map50_95 = []

        # Track last read position
        if start_fresh:
            # Start from end of file (skip existing data)
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    f.seek(0, 2)  # Seek to end of file
                    self.last_pos = f.tell()
                print(f"   Starting from position: {self.last_pos} (end of file)")
            except Exception as e:
                print(f"   Warning: Could not seek to end: {e}")
                self.last_pos = 0
        else:
            self.last_pos = 0

        # Track current epoch for data collection
        self.current_epoch = None
        self.epoch_data_collected = set()  # Track which epochs have data collected

        # Setup plots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Training Metrics', fontsize=16, fontweight='bold')

        # Configure subplots
        self.ax_losses = self.axes[0, 0]
        self.ax_val_losses = self.axes[0, 1]
        self.ax_metrics = self.axes[1, 0]
        self.ax_map = self.axes[1, 1]

        plt.tight_layout()

    def _find_latest_log(self):
        """Find the most recent train_model log file"""
        if not self.log_dir.exists():
            return None

        log_files = list(self.log_dir.glob("train_model_*.log"))
        if not log_files:
            return None

        # Sort by modification time, get latest
        latest = max(log_files, key=lambda p: p.stat().st_mtime)
        return latest

    def parse_log_line(self, line):
        """Parse a single log line and extract metrics"""
        # Match epoch number
        epoch_match = re.search(r'Epoch (\d+)/(\d+) completed', line)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            self.total_epochs = int(epoch_match.group(2))
            return

        # Only collect data if we have a current epoch
        if self.current_epoch is None:
            return

        # Skip if we've already collected data for this epoch
        if self.current_epoch in self.epoch_data_collected:
            return

        # Match training losses
        loss_match = re.search(r'Training losses - box: ([\d.]+), cls: ([\d.]+), dfl: ([\d.]+)', line)
        if loss_match:
            if self.current_epoch not in self.epochs:
                self.epochs.append(self.current_epoch)
                self.train_box_loss.append(float(loss_match.group(1)))
                self.train_cls_loss.append(float(loss_match.group(2)))
                self.train_dfl_loss.append(float(loss_match.group(3)))
            return

        # Match validation losses (collect all three together)
        if 'val/box_loss:' in line:
            val_box = re.search(r'val/box_loss: ([\d.]+)', line)
            if val_box and len(self.val_box_loss) < len(self.epochs):
                self.val_box_loss.append(float(val_box.group(1)))

        if 'val/cls_loss:' in line:
            val_cls = re.search(r'val/cls_loss: ([\d.]+)', line)
            if val_cls and len(self.val_cls_loss) < len(self.epochs):
                self.val_cls_loss.append(float(val_cls.group(1)))

        if 'val/dfl_loss:' in line:
            val_dfl = re.search(r'val/dfl_loss: ([\d.]+)', line)
            if val_dfl and len(self.val_dfl_loss) < len(self.epochs):
                self.val_dfl_loss.append(float(val_dfl.group(1)))

        # Match metrics
        if 'metrics/precision(B):' in line:
            prec = re.search(r'metrics/precision\(B\): ([\d.]+)', line)
            if prec and len(self.precision) < len(self.epochs):
                self.precision.append(float(prec.group(1)))

        if 'metrics/recall(B):' in line:
            rec = re.search(r'metrics/recall\(B\): ([\d.]+)', line)
            if rec and len(self.recall) < len(self.epochs):
                self.recall.append(float(rec.group(1)))

        if 'metrics/mAP50(B):' in line:
            map50 = re.search(r'metrics/mAP50\(B\): ([\d.]+)', line)
            if map50 and len(self.map50) < len(self.epochs):
                self.map50.append(float(map50.group(1)))

        if 'metrics/mAP50-95(B):' in line:
            map50_95 = re.search(r'metrics/mAP50-95\(B\): ([\d.]+)', line)
            if map50_95 and len(self.map50_95) < len(self.epochs):
                self.map50_95.append(float(map50_95.group(1)))
                # Mark epoch data as collected when we get the last metric
                self.epoch_data_collected.add(self.current_epoch)

    def read_new_lines(self):
        """Read new lines from log file since last read"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                # Seek to last position
                f.seek(self.last_pos)

                # Read new lines
                lines = f.readlines()

                # Update position
                self.last_pos = f.tell()

                # Parse new lines
                for line in lines:
                    self.parse_log_line(line)

                return len(lines) > 0
        except Exception as e:
            print(f"⚠️ Error reading log: {e}")
            return False

    def update_plots(self):
        """Update all plots with current data"""
        if not self.epochs:
            return

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Plot 1: Training Losses
        self.ax_losses.plot(self.epochs, self.train_box_loss, 'o-', label='Box Loss', linewidth=2)
        self.ax_losses.plot(self.epochs, self.train_cls_loss, 's-', label='Class Loss', linewidth=2)
        self.ax_losses.plot(self.epochs, self.train_dfl_loss, '^-', label='DFL Loss', linewidth=2)
        self.ax_losses.set_xlabel('Epoch', fontsize=12)
        self.ax_losses.set_ylabel('Loss', fontsize=12)
        self.ax_losses.set_title('Training Losses', fontsize=14, fontweight='bold')
        self.ax_losses.legend(loc='upper right')
        self.ax_losses.grid(True, alpha=0.3)

        # Plot 2: Validation Losses
        if len(self.val_box_loss) > 0:
            self.ax_val_losses.plot(self.epochs[:len(self.val_box_loss)], self.val_box_loss, 'o-', label='Val Box Loss', linewidth=2)
            self.ax_val_losses.plot(self.epochs[:len(self.val_cls_loss)], self.val_cls_loss, 's-', label='Val Class Loss', linewidth=2)
            self.ax_val_losses.plot(self.epochs[:len(self.val_dfl_loss)], self.val_dfl_loss, '^-', label='Val DFL Loss', linewidth=2)
        self.ax_val_losses.set_xlabel('Epoch', fontsize=12)
        self.ax_val_losses.set_ylabel('Loss', fontsize=12)
        self.ax_val_losses.set_title('Validation Losses', fontsize=14, fontweight='bold')
        self.ax_val_losses.legend(loc='upper right')
        self.ax_val_losses.grid(True, alpha=0.3)

        # Plot 3: Precision & Recall
        if len(self.precision) > 0:
            self.ax_metrics.plot(self.epochs[:len(self.precision)], self.precision, 'o-', label='Precision', linewidth=2, color='green')
            self.ax_metrics.plot(self.epochs[:len(self.recall)], self.recall, 's-', label='Recall', linewidth=2, color='orange')
        self.ax_metrics.set_xlabel('Epoch', fontsize=12)
        self.ax_metrics.set_ylabel('Score', fontsize=12)
        self.ax_metrics.set_title('Precision & Recall', fontsize=14, fontweight='bold')
        self.ax_metrics.legend(loc='lower right')
        self.ax_metrics.grid(True, alpha=0.3)
        self.ax_metrics.set_ylim([0, 1])

        # Plot 4: mAP Scores
        if len(self.map50) > 0:
            self.ax_map.plot(self.epochs[:len(self.map50)], self.map50, 'o-', label='mAP@50', linewidth=2, color='blue')
            self.ax_map.plot(self.epochs[:len(self.map50_95)], self.map50_95, 's-', label='mAP@50-95', linewidth=2, color='red')
        self.ax_map.set_xlabel('Epoch', fontsize=12)
        self.ax_map.set_ylabel('mAP', fontsize=12)
        self.ax_map.set_title('Mean Average Precision', fontsize=14, fontweight='bold')
        self.ax_map.legend(loc='lower right')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_ylim([0, 1])

        # Update figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def print_summary(self):
        """Print current training summary"""
        if not self.epochs:
            return

        latest_epoch = self.epochs[-1]
        total = self.total_epochs if hasattr(self, 'total_epochs') else '?'

        print(f"\n{'='*60}")
        print(f"📊 Latest Metrics (Epoch {latest_epoch}/{total})")
        print(f"{'='*60}")

        if self.train_box_loss:
            print(f"   Train Losses:")
            print(f"      Box:   {self.train_box_loss[-1]:.4f}")
            print(f"      Class: {self.train_cls_loss[-1]:.4f}")
            print(f"      DFL:   {self.train_dfl_loss[-1]:.4f}")

        if self.val_box_loss:
            print(f"   Val Losses:")
            print(f"      Box:   {self.val_box_loss[-1]:.4f}")
            print(f"      Class: {self.val_cls_loss[-1]:.4f}")
            print(f"      DFL:   {self.val_dfl_loss[-1]:.4f}")

        if self.precision:
            print(f"   Metrics:")
            print(f"      Precision: {self.precision[-1]:.4f}")
            print(f"      Recall:    {self.recall[-1]:.4f}")
            print(f"      mAP@50:    {self.map50[-1]:.4f}")
            print(f"      mAP@50-95: {self.map50_95[-1]:.4f}")

        print(f"{'='*60}\n")

    def run(self, update_interval=2.0):
        """Main monitoring loop"""
        print(f"\n{'='*60}")
        print(f"🚀 Real-time Training Monitor Started")
        print(f"{'='*60}")
        print(f"   Log file: {self.log_file.name}")
        print(f"   Update interval: {update_interval}s")
        print(f"   Press Ctrl+C to stop")
        print(f"{'='*60}\n")

        # Initial read
        self.read_new_lines()
        self.update_plots()

        try:
            last_update = time.time()
            last_epoch_count = 0

            while True:
                # Read new log lines
                has_new_data = self.read_new_lines()

                current_time = time.time()

                # Update plots periodically or when new data arrives
                if has_new_data or (current_time - last_update) >= update_interval:
                    self.update_plots()

                    # Print summary when new epoch completed
                    if len(self.epochs) > last_epoch_count:
                        self.print_summary()
                        last_epoch_count = len(self.epochs)

                    last_update = current_time

                # Sleep briefly to avoid excessive CPU usage
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            print(f"   Total epochs monitored: {len(self.epochs)}")

            # Keep plot window open
            print("\n💡 Tip: Close the plot window to exit completely")
            plt.ioff()
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Real-time Training Monitor')
    parser.add_argument('--log-file', type=str, help='Path to log file (auto-detects latest if not specified)')
    parser.add_argument('--interval', type=float, default=10.0, help='Update interval in seconds (default: 10.0)')
    parser.add_argument('--fresh', action='store_true', help='Start monitoring from current position (skip existing log data)')

    args = parser.parse_args()

    monitor = TrainingMonitor(log_file=args.log_file, start_fresh=args.fresh)
    monitor.run(update_interval=args.interval)

if __name__ == "__main__":
    main()
