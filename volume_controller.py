import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math


class VolumeController:
    """
    Volume Controller class that manages system volume based on finger distance
    Uses Windows Core Audio APIs through pycaw
    """
    
    def __init__(self, min_distance=30, max_distance=300):
        """
        Initialize Volume Controller
        
        Args:
            min_distance: Minimum finger distance for volume calculation
            max_distance: Maximum finger distance for volume calculation
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        # Initialize Windows Core Audio API
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Get volume range
            self.vol_range = self.volume.GetVolumeRange()
            self.min_vol = self.vol_range[0]  # -65.25 dB
            self.max_vol = self.vol_range[1]  # 0.0 dB
            
            print(f"Volume Controller initialized successfully!")
            print(f"Volume range: {self.min_vol} dB to {self.max_vol} dB")
            
        except Exception as e:
            print(f"Error initializing volume controller: {e}")
            self.volume = None
    
    def map_distance_to_volume(self, distance):
        """
        Map finger distance to volume level
        
        Args:
            distance: Distance between thumb and index finger
            
        Returns:
            volume_db: Volume level in decibels
            volume_percent: Volume level as percentage (0-100)
        """
        # Clamp distance to valid range
        distance = max(self.min_distance, min(distance, self.max_distance))
        
        # Map distance to volume percentage (0-100)
        volume_percent = np.interp(distance, 
                                 [self.min_distance, self.max_distance], 
                                 [0, 100])
        
        # Map percentage to decibel range
        volume_db = np.interp(volume_percent, 
                            [0, 100], 
                            [self.min_vol, self.max_vol])
        
        return volume_db, volume_percent
    
    def set_volume(self, distance):
        """
        Set system volume based on finger distance
        
        Args:
            distance: Distance between thumb and index finger
            
        Returns:
            success: Boolean indicating if volume was set successfully
            volume_percent: Current volume percentage
        """
        if self.volume is None:
            return False, 0
        
        try:
            volume_db, volume_percent = self.map_distance_to_volume(distance)
            
            # Set the volume
            self.volume.SetMasterVolumeLevel(volume_db, None)
            
            return True, int(volume_percent)
            
        except Exception as e:
            print(f"Error setting volume: {e}")
            return False, 0
    
    def get_current_volume(self):
        """
        Get current system volume
        
        Returns:
            volume_percent: Current volume as percentage (0-100)
        """
        if self.volume is None:
            return 0
        
        try:
            current_db = self.volume.GetMasterVolumeLevel()
            volume_percent = np.interp(current_db, 
                                     [self.min_vol, self.max_vol], 
                                     [0, 100])
            return int(volume_percent)
        
        except Exception as e:
            print(f"Error getting volume: {e}")
            return 0
    
    def mute_volume(self):
        """Mute the system volume"""
        if self.volume is not None:
            try:
                self.volume.SetMute(1, None)
                return True
            except Exception as e:
                print(f"Error muting volume: {e}")
                return False
        return False
    
    def unmute_volume(self):
        """Unmute the system volume"""
        if self.volume is not None:
            try:
                self.volume.SetMute(0, None)
                return True
            except Exception as e:
                print(f"Error unmuting volume: {e}")
                return False
        return False
    
    def is_muted(self):
        """Check if volume is muted"""
        if self.volume is not None:
            try:
                return bool(self.volume.GetMute())
            except Exception as e:
                print(f"Error checking mute status: {e}")
                return False
        return False
    
    def calculate_volume_smoothing(self, distances, window_size=5):
        """
        Apply smoothing to volume changes to reduce jitter
        
        Args:
            distances: List of recent distance measurements
            window_size: Number of measurements to average
            
        Returns:
            smoothed_distance: Averaged distance value
        """
        if len(distances) < window_size:
            return np.mean(distances) if distances else 0
        
        # Use moving average for smoothing
        recent_distances = distances[-window_size:]
        return np.mean(recent_distances)
    
    def get_volume_bar_coordinates(self, img_height, img_width, volume_percent):
        """
        Calculate coordinates for volume bar visualization
        
        Args:
            img_height: Image height
            img_width: Image width
            volume_percent: Volume percentage (0-100)
            
        Returns:
            bar_coords: Dictionary with bar coordinates
        """
        bar_height = int(img_height * 0.6)  # 60% of image height
        bar_width = 50
        bar_x = img_width - bar_width - 20  # Right side with margin
        bar_y = int((img_height - bar_height) / 2)  # Centered vertically
        
        # Calculate filled portion based on volume
        filled_height = int((volume_percent / 100) * bar_height)
        
        return {
            'bar_x': bar_x,
            'bar_y': bar_y,
            'bar_width': bar_width,
            'bar_height': bar_height,
            'filled_height': filled_height,
            'filled_y': bar_y + bar_height - filled_height
        } 