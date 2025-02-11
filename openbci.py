#! /usr/bin/env python3

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import numpy as np
import time
import os
import csv
from pyautogui import press
from dataclasses import dataclass
# Updated import statement
from fuzzy_controller import TrainableFuzzyController, FuzzyParameters

class NeuralOscillations():
    def __init__(self, params_file=None, sound_duration=0.5, sampling_rate=44100, timeout=0, 
                 board_id=BoardIds.SYNTHETIC_BOARD, ip_port=0, ip_protocol=0, ip_address='', 
                 serial_port='', mac_address='', streamer_params='', serial_number='', 
                 file='', master_board=BoardIds.NO_BOARD):
        
        self.sound_duration = sound_duration
        self.sampling_rate = sampling_rate
        self.timeout = timeout
        self.board_id = board_id
        self.ip_port = ip_port
        self.ip_protocol = ip_protocol
        self.ip_address = ip_address
        self.serial_port = serial_port
        self.mac_address = mac_address
        self.streamer_params = streamer_params
        self.serial_number = serial_number
        self.file = file
        self.master_board = master_board

        self.wf = 3  # BLACKMAN_HARRIS Window Function
        self.num_samples = 512
        
        # Initialize controller
        if params_file and os.path.exists(params_file):
            self.controller = TrainableFuzzyController.load_parameters(params_file)
            print(f"Loaded controller parameters from {params_file}")
        else:
            self.controller = TrainableFuzzyController()
            print("Using default controller parameters")

    def adjust_youtube_speed(self, target_speed):
        """
        Adjust YouTube playback speed using keyboard shortcuts
        Returns the new current speed
        """
        # Calculate number of steps needed
        steps = round((target_speed - self.controller.current_speed) / 
                     self.controller.config.step_size)
        
        # Apply speed changes
        if steps > 0:
            # Speed up
            for _ in range(abs(steps)):
                press(['>'])
            self.controller.current_speed += steps * self.controller.config.step_size
        elif steps < 0:
            # Slow down
            for _ in range(abs(steps)):
                press(['<'])
            self.controller.current_speed += steps * self.controller.config.step_size
        
        # Ensure speed stays within bounds
        self.controller.current_speed = np.clip(
            self.controller.current_speed,
            self.controller.config.min_speed,
            self.controller.config.max_speed
        )
        
        return self.controller.current_speed

    def initialise_board(self):
        params = BrainFlowInputParams()
        params.timeout = self.timeout
        params.board_id = self.board_id
        params.ip_port = self.ip_port
        params.ip_protocol = self.ip_protocol
        params.ip_address = self.ip_address
        params.serial_port = self.serial_port
        params.mac_address = self.mac_address
        params.streamer_params = self.streamer_params
        params.serial_number = self.serial_number
        params.file = self.file
        params.master_board = self.master_board
        return BoardShim(self.board_id, params)

    def filter(self, eeg_data, lower_bound, upper_bound, window_function):
        psd = DataFilter.get_psd(eeg_data, BoardShim.get_sampling_rate(self.board_id), window_function)
        return DataFilter.get_band_power(psd, lower_bound, upper_bound)

    def eeg_recorder(self, eeg_channel_count=8, mode='control'):
        """
        Main recording loop with both control and training modes
        
        Args:
            eeg_channel_count (int): Number of EEG channels to process
            mode (str): Either 'control' for real-time control or 'training' for data collection
        """
        board = self.initialise_board()
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        board.prepare_session()
        board.start_stream()
        time.sleep(5)  # Increased stabilization time to 5 seconds
        
        # For training mode
        if mode == 'training':
            theta_pow = []
            alpha_pow = []
            beta_pow = []
            gamma_pow = []
            ta_ratios = []
            tb_ratios = []
            tg_ratios = []
            ab_ratios = []
            ag_ratios = []
            bg_ratios = []
        
        print(f"Starting EEG recording in {mode} mode...")
        if mode == 'control':
            print("Current speed:", self.controller.current_speed)
        
        try:
            while True:
                data = board.get_current_board_data(self.num_samples)
                tMean, aMean, bMean, gMean = 0, 0, 0, 0
                
                # Calculate band powers across channels
                for c in range(eeg_channel_count):
                    tMean += np.mean(self.filter(data[eeg_channels[c]], 4, 8, self.wf))
                    aMean += np.mean(self.filter(data[eeg_channels[c]], 8, 13, self.wf))
                    bMean += np.mean(self.filter(data[eeg_channels[c]], 13, 32, self.wf))
                    gMean += np.mean(self.filter(data[eeg_channels[c]], 32, 100, self.wf))
                
                # Average across channels
                tMean /= eeg_channel_count
                aMean /= eeg_channel_count
                bMean /= eeg_channel_count
                gMean /= eeg_channel_count
                
                if mode == 'control':
                    # Get target speed from controller
                    target_speed = self.controller.update(tMean, aMean, bMean)
                    
                    # Adjust YouTube speed
                    current_speed = self.adjust_youtube_speed(target_speed)
                    
                    # Print status
                    print(f"\rθ: {tMean:.2f} α: {aMean:.2f} β: {bMean:.2f} "
                          f"Target: {target_speed:.2f}x Current: {current_speed:.2f}x", 
                          end="")
                
                else:  # Training mode
                    # Calculate ratios
                    tar = tMean / aMean if aMean > 0 else 0
                    tbr = tMean / bMean if bMean > 0 else 0
                    tgr = tMean / gMean if gMean > 0 else 0
                    abr = aMean / bMean if bMean > 0 else 0
                    agr = aMean / gMean if gMean > 0 else 0
                    bgr = bMean / gMean if gMean > 0 else 0
                    
                    # Store data
                    theta_pow.append(tMean)
                    alpha_pow.append(aMean)
                    beta_pow.append(bMean)
                    gamma_pow.append(gMean)
                    ta_ratios.append(tar)
                    tb_ratios.append(tbr)
                    tg_ratios.append(tgr)
                    ab_ratios.append(abr)
                    ag_ratios.append(agr)
                    bg_ratios.append(bgr)
                    
                    # Print current values
                    print(f"\rθ: {tMean:.2f} α: {aMean:.2f} β: {bMean:.2f} γ: {gMean:.2f} "
                          f"TAR: {tar:.2f} TBR: {tbr:.2f}", end="")
                
                time.sleep(1)  # Update interval
                
        except KeyboardInterrupt:
            print("\nStopping recording...")
            if mode == 'training':
                self.create_csv(theta_pow, alpha_pow, beta_pow, gamma_pow,
                              ta_ratios, tb_ratios, tg_ratios, ab_ratios,
                              ag_ratios, bg_ratios)
            board.stop_stream()
            board.release_session()
            raise Exception

    def create_csv(self, *data):
        """Create CSV file with recorded data"""
        try:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"output_{timestamp}.csv"
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archive')
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path, filename)
            
            rows = zip(*data)
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ["Theta_pow", "Alpha_pow", "Beta_pow", "Gamma_pow",
                         "TAR", "TBR", "TGR", "ABR", "AGR", "BGR"]
                writer.writerow(header)
                writer.writerows(rows)
            print(f"\nData has been written to {file_path}")
        except Exception as e:
            print(f"Error creating CSV: {e}")

# Board Details
BOARD_ID = 0  # CYTON_BOARD
EEG_CHANNEL_COUNT = 8
# Port Details
SERIAL_PORT = "COM3"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenBCI EEG Recording and Control')
    parser.add_argument('--mode', type=str, choices=['control', 'training'],
                      default='control', help='Operation mode: control or training')
    parser.add_argument('--params', type=str, default=None,
                      help='Path to controller parameters JSON file')
    args = parser.parse_args()
    
    neural_oscillations = NeuralOscillations(
        board_id=BOARD_ID,
        serial_port=SERIAL_PORT,
        params_file=args.params
    )
    
    neural_oscillations.eeg_recorder(
        eeg_channel_count=EEG_CHANNEL_COUNT,
        mode=args.mode
    )