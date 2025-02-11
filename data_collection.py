#! /usr/bin/env python3

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import numpy as np
import time
import os
import csv
import json
from datetime import datetime
import keyboard

class DataCollector:
    def __init__(self, board_id=0, serial_port='COM3'):
        self.board_id = board_id
        self.serial_port = serial_port
        self.wf = 3  # BLACKMAN_HARRIS Window Function
        self.num_samples = 512
        self.session_data = {
            'metadata': {},
            'recordings': []
        }
    
    def initialize_board(self):
        params = BrainFlowInputParams()
        params.board_id = self.board_id
        params.serial_port = self.serial_port
        board = BoardShim(self.board_id, params)
        return board
    
    def filter(self, eeg_data, lower_bound, upper_bound, window_function):
        psd = DataFilter.get_psd(eeg_data, BoardShim.get_sampling_rate(self.board_id), window_function)
        return DataFilter.get_band_power(psd, lower_bound, upper_bound)
    
    def collect_baseline(self, duration=60):
        """Collect baseline data while user is in a relaxed state"""
        print("\nBaseline Recording")
        print("==================")
        print("We'll record your baseline brain activity.")
        print("Please:")
        print("- Sit comfortably")
        print("- Keep your eyes open")
        print("- Try to stay relaxed")
        print("- Avoid any mentally demanding tasks")
        input("\nPress Enter when you're ready to start baseline recording...")
        
        self._record_session('baseline', duration)
    
    def collect_high_load(self, duration=300):
        """Collect data during high cognitive load"""
        print("\nHigh Cognitive Load Recording")
        print("============================")
        print("Now we'll record during a mentally demanding task.")
        print("Suggested activities:")
        print("- Watch a technical lecture video")
        print("- Read a complex academic paper")
        print("- Solve mathematical problems")
        print("- Study new material")
        input("\nPress Enter when you're ready to start high load recording...")
        
        self._record_session('high_load', duration)
    
    def collect_low_load(self, duration=300):
        """Collect data during low cognitive load"""
        print("\nLow Cognitive Load Recording")
        print("===========================")
        print("Now we'll record during a less demanding task.")
        print("Suggested activities:")
        print("- Watch an entertainment video")
        print("- Read casual content")
        print("- Browse social media")
        input("\nPress Enter when you're ready to start low load recording...")
        
        self._record_session('low_load', duration)
    
    def _record_session(self, session_type, duration):
        board = self.initialize_board()
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
        theta_pow = []
        alpha_pow = []
        beta_pow = []
        gamma_pow = []
        timestamps = []
        
        try:
            board.prepare_session()
            board.start_stream()
            print("\nStabilizing signal...", end='', flush=True)
            time.sleep(5)  # Wait for signal to stabilize
            print("done")
            
            start_time = time.time()
            print(f"\nRecording started. Duration: {duration} seconds")
            print("Press 'q' to stop recording early if needed")
            print("\nCurrent readings:")
            
            while (time.time() - start_time) < duration:
                if keyboard.is_pressed('q'):
                    print("\nRecording stopped early by user")
                    break
                
                elapsed = int(time.time() - start_time)
                remaining = duration - elapsed
                
                data = board.get_current_board_data(self.num_samples)
                tMean, aMean, bMean, gMean = 0, 0, 0, 0
                
                for c in range(len(eeg_channels)):
                    tMean += np.mean(self.filter(data[eeg_channels[c]], 4, 8, self.wf))
                    aMean += np.mean(self.filter(data[eeg_channels[c]], 8, 13, self.wf))
                    bMean += np.mean(self.filter(data[eeg_channels[c]], 13, 32, self.wf))
                    gMean += np.mean(self.filter(data[eeg_channels[c]], 32, 100, self.wf))
                
                channel_count = len(eeg_channels)
                tMean /= channel_count
                aMean /= channel_count
                bMean /= channel_count
                gMean /= channel_count
                
                theta_pow.append(tMean)
                alpha_pow.append(aMean)
                beta_pow.append(bMean)
                gamma_pow.append(gMean)
                timestamps.append(elapsed)
                
                print(f"\rTime remaining: {remaining}s | θ: {tMean:.2f} α: {aMean:.2f} "
                      f"β: {bMean:.2f} γ: {gMean:.2f}", end='', flush=True)
                
                time.sleep(1)
            
            print("\n\nRecording complete!")
            
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            board.stop_stream()
            board.release_session()
        
        session_data = {
            'type': session_type,
            'duration': elapsed,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'theta_pow': theta_pow,
                'alpha_pow': alpha_pow,
                'beta_pow': beta_pow,
                'gamma_pow': gamma_pow,
                'timestamps': timestamps
            }
        }
        
        self.session_data['recordings'].append(session_data)
    
    def save_data(self):
        """Save all collected data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
        os.makedirs(base_path, exist_ok=True)
        
        # Save raw data as JSON
        json_path = os.path.join(base_path, f'eeg_data_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.session_data, f, indent=4)
        
        # Save processed CSV files for each session
        for recording in self.session_data['recordings']:
            session_type = recording['type']
            data = recording['data']
            
            csv_path = os.path.join(base_path, f'{session_type}_{timestamp}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Theta_pow", "Alpha_pow", "Beta_pow", "Gamma_pow"])
                
                for i in range(len(data['timestamps'])):
                    writer.writerow([
                        data['timestamps'][i],
                        data['theta_pow'][i],
                        data['alpha_pow'][i],
                        data['beta_pow'][i],
                        data['gamma_pow'][i]
                    ])
        
        print(f"\nData saved in {base_path}")
        print(f"JSON data: {json_path}")
        print("CSV files:")
        print(f"- {base_path}/*.csv")

def main():
    print("EEG Data Collection for Controller Training")
    print("==========================================")
    print("\nThis script will collect EEG data in three phases:")
    print("1. Baseline (1 minute)")
    print("2. High cognitive load (5 minutes)")
    print("3. Low cognitive load (5 minutes)")
    
    board_id = input("\nEnter board ID [0 for Cyton]: ") or "0"
    serial_port = input("Enter serial port [COM3]: ") or "COM3"
    
    collector = DataCollector(
        board_id=int(board_id),
        serial_port=serial_port
    )
    
    try:
        collector.collect_baseline()
        collector.collect_high_load()
        collector.collect_low_load()
        collector.save_data()
        
        print("\nData collection complete!")
        print("You can now use this data to train your controller.")
        
    except Exception as e:
        print(f"\nError during data collection: {e}")
        raise

if __name__ == "__main__":
    main()