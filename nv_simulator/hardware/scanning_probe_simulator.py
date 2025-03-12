# -*- coding: utf-8 -*-
"""
This file contains the Qudi simulator module for a confocal scanner with NV centers.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

from logging import getLogger
import time
from typing import Optional, Dict, Tuple, Any, List
import numpy as np
from PySide2 import QtCore
from fysom import FysomError
from qudi.core.configoption import ConfigOption
from qudi.util.mutex import RecursiveMutex
from qudi.util.constraints import ScalarConstraint
from qudi.interface.scanning_probe_interface import (
    ScanningProbeInterface,
    ScanData,
    ScanConstraints,
    ScannerAxis,
    ScannerChannel,
    ScanSettings,
    BackScanCapability,
    CoordinateTransformMixin,
)
from dataclasses import dataclass
import os
import sys
import threading
import random

# Add the nv_simulator directory to the path so we can import the simulator
script_dir = os.path.dirname(os.path.abspath(__file__))
simulator_dir = os.path.dirname(script_dir)
if simulator_dir not in sys.path:
    sys.path.append(simulator_dir)

from nv_simulator import NVCenterSimulator

logger = getLogger(__name__)


@dataclass(frozen=True)
class SimulatorScanConstraints(ScanConstraints):
    """Extended scan constraints for the NV simulator."""
    spot_number: ScalarConstraint


class ScanningProbeSimulator(ScanningProbeInterface):
    """
    A scanning probe simulator that generates synthetic confocal images with simulated NV centers.
    Connects to the NV center simulator for detailed spin dynamics.

    Example config for copy-paste:

    scanning_probe_simulator:
        module.Class: 'nv_simulator.hardware.scanning_probe_simulator.ScanningProbeSimulator'
        options:
            max_scanner_update_rate: 20  # Hz
            external_clock_rate: 10  # Hz
            backscan_configurable: True
            position_ranges:
                x: [-50, 50]  # μm
                y: [-50, 50]  # μm
                z: [-25, 25]  # μm
            spot_number: 15
            nv_density: 0.01  # NV centers per μm²
    """

    # Config options
    _max_scanner_update_rate = ConfigOption('max_scanner_update_rate', 20)  # Hz
    _backscan_configurable = ConfigOption('backscan_configurable', True)
    _position_ranges = ConfigOption('position_ranges', dict(x=[-100, 100], y=[-100, 100], z=[-10, 10]))
    _spot_number = ConfigOption('spot_number', 10)
    _external_clock_rate = ConfigOption('external_clock_rate', 10)  # Hz
    _nv_density = ConfigOption('nv_density', 0.01)  # Centers per μm²

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Scanning state variables
        self._current_position = None  # Dict[str, float]
        self._target_position = None  # Dict[str, float]
        self._scanner_axes = None  # Dict[str, ScannerAxis]
        self._scanner_channels = None  # Dict[str, ScannerChannel]
        self._scan_settings = None  # ScanSettings
        
        # Scan data
        self._created_data_channels = None
        self._current_scan_data = None
        self._current_setups = None
        
        # Threading
        self._stop_requested = False
        self._scanning = False
        self._scan_loop_thread = None
        self._mutex = RecursiveMutex()
        self._state_machine_mutex = RecursiveMutex()

        # NV center simulation
        self._nv_simulator = None
        self._nv_centers = []  # List of NV center positions and properties
        
        # Timer for mock frequency-scan
        self._scanner_position_timer = QtCore.QTimer()
        self._scanner_position_timer.setSingleShot(False)
        self._scanner_position_timer.timeout.connect(self._set_next_scanner_position)

    def on_activate(self):
        """Initialize the scanner interface."""
        try:
            # Initialize the NV center simulator
            self._nv_simulator = NVCenterSimulator()
            
            # Generate scanner axes
            self._scanner_axes = dict()
            for name, range_values in self._position_ranges.items():
                if len(name) <= 0:
                    raise ValueError(f'Failed to parse scanner axis "{name}". '
                                     f'Must be non-empty string.')
                
                # Configure axis
                min_val, max_val = tuple(range_values)
                if min_val >= max_val:
                    raise ValueError(f'Scanner axis "{name}" has invalid scan range: [{min_val}, {max_val}]')
                
                # Create scanner axis with properties 
                step_constr = ScalarConstraint(0, (0, max_val - min_val))
                self._scanner_axes[name] = ScannerAxis(
                    name=name,
                    unit='μm',
                    position=ScalarConstraint(min_val, (min_val, max_val)),
                    step=step_constr,
                    resolution=ScalarConstraint(100, (10, 1000), enforce_int=True),
                    frequency=ScalarConstraint(self._external_clock_rate, 
                                               (0.1, self._max_scanner_update_rate))
                )
            
            # Set initial position to center of the range
            self._current_position = dict()
            for name, axis in self._scanner_axes.items():
                min_val, max_val = axis.position.bounds
                self._current_position[name] = (min_val + max_val) / 2
            
            # The target position is initially the current position
            self._target_position = self._current_position.copy()
            
            # Configure channels
            self._scanner_channels = {
                'count_rate': ScannerChannel(name='count_rate', unit='c/s')
            }
            
            # Create simulated NV centers with random positions and properties
            self._create_random_nv_centers()
            
            # Set up scanning state
            self._scanning = False
            self._created_data_channels = None
            self._current_scan_data = None
            self._current_setups = None
            
            # Configure scanner timer
            self._scanner_position_timer.setInterval(int(1000 / self._max_scanner_update_rate))
            
        except Exception as exc:
            self.log.exception("Error during scanning probe simulator activation.")
            raise exc

    def on_deactivate(self):
        """Stop any running processes and de-initialize hardware."""
        try:
            # Stop scanner position timer if running
            if self._scanner_position_timer.isActive():
                self._scanner_position_timer.stop()
                
            # Stop any running scan loops
            if self._scanning:
                self._stop_scan()
                
            # Clean up NV simulator
            self._nv_simulator = None
            
        except Exception as exc:
            self.log.exception("Error during scanning probe simulator deactivation.")
            raise exc

    @property
    def scanner_axes(self) -> Dict[str, ScannerAxis]:
        """Dictionary of scanner axes metadata."""
        with self._mutex:
            return self._scanner_axes.copy()

    @property
    def scanner_channels(self) -> Dict[str, ScannerChannel]:
        """Dictionary of scanner channel metadata."""
        with self._mutex:
            return self._scanner_channels.copy()

    @property
    def scanner_constraints(self) -> SimulatorScanConstraints:
        """Scanner constraints for this hardware."""
        with self._mutex:
            # Combine all axes
            axes = tuple(self._scanner_axes.values())
            channels = tuple(self._scanner_channels.values())
            
            # Create constraints object
            back_caps = BackScanCapability(0)
            if self._backscan_configurable:
                back_caps |= BackScanCapability.AVAILABLE | BackScanCapability.FULLY_CONFIGURABLE
            
            # Add custom spot number constraint
            spot_constr = ScalarConstraint(default=self._spot_number, bounds=(1, 100), enforce_int=True)
            
            return SimulatorScanConstraints(
                channel_objects=channels,
                axis_objects=axes,
                back_scan_capability=back_caps,
                has_position_feedback=False,
                square_px_only=False,
                spot_number=spot_constr
            )

    @property
    def scanner_position(self) -> Dict[str, float]:
        """Current scanner position in axis units. 
        The axis keys correspond to scanner_axes.
        """
        with self._mutex:
            return self._current_position.copy()

    @property
    def target_position(self) -> Dict[str, float]:
        """Target scanner position based on move_scanner command."""
        with self._mutex:
            return self._target_position.copy()

    def move_scanner(self, position: Dict[str, float]) -> None:
        """Move scanner to a given position in axis units."""
        with self._mutex:
            # Check if all provided axes are valid
            for axis in position:
                if axis not in self._scanner_axes:
                    raise ValueError(f'Unknown scanner axis "{axis}" encountered.')
            
            # Check if positions are within bounds
            for axis, pos in position.items():
                constr = self._scanner_axes[axis].position
                if not constr.is_valid(pos)[0]:
                    min_val, max_val = constr.bounds
                    raise ValueError(
                        f'Scanner position for axis "{axis}" ({pos}) must be in range [{min_val}, {max_val}].'
                    )
            
            # Set target position (partial updates allowed)
            for axis, pos in position.items():
                self._target_position[axis] = pos
            
            # Simulate movement
            self._move_scanner_to_target()

    def _move_scanner_to_target(self):
        """Simulate scanner movement to target position."""
        # In a real system, this would command the actual hardware
        # Here we just update position immediately
        for axis, target in self._target_position.items():
            self._current_position[axis] = target

    def configure_scan(self, settings: ScanSettings) -> Tuple[ScanSettings, ScanSettings]:
        """Configure scanner for a scan with the given settings."""
        with self._mutex:
            # Get constraints and validate settings
            constraints = self.scanner_constraints
            constraints.check_settings(settings)
            
            # Create back settings if needed
            back_settings = None
            if BackScanCapability.AVAILABLE in constraints.back_scan_capability:
                back_settings = settings
            
            return settings, back_settings

    def start_scan(self, settings: ScanSettings, back_settings: Optional[ScanSettings] = None) -> None:
        """Start a new scan with the given settings."""
        with self._state_machine_mutex:
            if self.module_state() != 'idle':
                raise RuntimeError('Start scan failed. Scanning already in progress.')
            self.module_state.lock()
        
        try:
            with self._mutex:
                # Apply settings
                self._scan_settings = settings
                
                # Create data containers
                self._created_data_channels = {}
                channel_data = {}
                
                # Calculate array shapes for scan data
                resolution = tuple(int(res) for res in settings.resolution)
                for channel_name in settings.channels:
                    if channel_name not in self._scanner_channels:
                        raise ValueError(f'Unknown scanner channel "{channel_name}" encountered.')
                    
                    # Create empty data array
                    dtype = np.dtype(self._scanner_channels[channel_name].dtype)
                    array = np.zeros(resolution, dtype=dtype)
                    channel_data[channel_name] = array
                
                # Create scan data container
                axes_names = tuple(settings.axes)
                axes_units = tuple(self._scanner_axes[name].unit for name in axes_names)
                self._current_scan_data = ScanData(
                    channels=channel_data,
                    channel_units={name: self._scanner_channels[name].unit for name in settings.channels},
                    axes=axes_names,
                    axes_units=axes_units,
                    scan_range=settings.range,
                )
                
                # Store settings for scan loop
                self._current_setups = (settings, back_settings)
                
                # Start scan loop in a new thread
                self._scanning = True
                self._stop_requested = False
                self._scan_loop_thread = threading.Thread(target=self._scan_loop)
                self._scan_loop_thread.daemon = True
                self._scan_loop_thread.start()
                
                # Start position timer for hardware scan (only needed for scanner position updates)
                self._scanner_position_timer.start()
                
        except Exception as exc:
            self.module_state.unlock()
            self.log.exception('Error while starting scan:')
            raise exc

    def stop_scan(self) -> None:
        """Stop the currently running scan."""
        with self._state_machine_mutex:
            if self.module_state() == 'idle':
                self.log.warning('Stop scan called while not scanning')
                return
        
        try:
            self._stop_scan()
        except Exception as exc:
            self.log.exception('Error while stopping scan:')
            raise exc

    def _stop_scan(self):
        """Internal method to stop a scan."""
        with self._mutex:
            # Set stop flag for scan loop
            self._stop_requested = True
            
            # Stop scanner position timer
            if self._scanner_position_timer.isActive():
                self._scanner_position_timer.stop()
                
            # Wait for scan loop to finish
            if self._scan_loop_thread is not None and self._scan_loop_thread.is_alive():
                self._scan_loop_thread.join(timeout=5.0)
            
            # Clean up
            self._scanning = False
            self.module_state.unlock()

    def current_scan_data(self) -> Optional[ScanData]:
        """Returns the current scan data. May return None if no scan is running or no data has been acquired yet."""
        with self._mutex:
            return self._current_scan_data

    def _scan_loop(self):
        """Main scan loop that simulates data acquisition."""
        try:
            settings, back_settings = self._current_setups
            axes_names = settings.axes
            
            # Calculate step sizes for each axis
            step_sizes = []
            for i, axis_name in enumerate(axes_names):
                start, stop = settings.range[i]
                res = settings.resolution[i]
                step_sizes.append((stop - start) / (res - 1) if res > 1 else 0)
            
            # Calculate time per pixel
            pixel_time = 1.0 / settings.frequency  # seconds
            
            # Iterate over all scan points
            shape = tuple(int(res) for res in settings.resolution)
            total_points = np.prod(shape)
            
            for point_idx in range(total_points):
                # Check if stop was requested
                if self._stop_requested:
                    break
                
                # Convert linear index to n-dimensional indices
                indices = np.unravel_index(point_idx, shape)
                
                # Calculate positions for each axis
                positions = {}
                for i, axis_name in enumerate(axes_names):
                    start, _ = settings.range[i]
                    positions[axis_name] = start + indices[i] * step_sizes[i]
                
                # Move scanner to position
                self.move_scanner(positions)
                
                # Simulate data acquisition
                for channel_name in settings.channels:
                    # Get channel data
                    data = self._current_scan_data.channels[channel_name]
                    
                    # Simulate photon counts at this position
                    count_rate = self._simulate_count_rate(positions)
                    
                    # Add noise
                    count_rate += np.random.normal(0, count_rate * 0.05)
                    
                    # Store value
                    data[indices] = max(0, count_rate)
                
                # Signal data update
                self.sigUpdateScanData.emit()
                
                # Wait for next pixel
                time.sleep(pixel_time)
            
            # Scan completed successfully
            self.log.debug('Scan completed')
            
        except Exception as exc:
            self.log.exception('Error in scan loop:')
        finally:
            # Make sure the scanning state is properly reset
            with self._mutex:
                if not self._stop_requested:
                    self._stop_requested = True
                    self._scanning = False
                    with self._state_machine_mutex:
                        if self.module_state() != 'idle':
                            self.module_state.unlock()

    def _simulate_count_rate(self, position):
        """Simulate photon count rate for a given scanner position."""
        # Calculate distance to each NV center
        count_rate = 0
        background = 1000  # Background counts per second
        
        for nv in self._nv_centers:
            # Calculate squared distance to NV center
            distance_sq = 0
            for axis, pos in position.items():
                if axis in nv['position']:
                    distance_sq += (pos - nv['position'][axis])**2
            
            # Convert to distance in μm
            distance = np.sqrt(distance_sq)
            
            # Point spread function (PSF) of the confocal microscope
            # Approximate as Gaussian with FWHM of 300 nm
            sigma = 0.3 / 2.355  # Convert FWHM to sigma (in μm)
            psf = np.exp(-0.5 * (distance / sigma)**2)
            
            # NV center brightness depends on its properties
            brightness = nv['brightness'] * 100000  # Max counts per second
            
            # Add contribution from this NV center
            count_rate += brightness * psf
        
        # Add background counts and noise
        count_rate += background + np.random.poisson(np.sqrt(count_rate + background))
        
        return count_rate

    def _set_next_scanner_position(self):
        """Update scanner position for ongoing movement simulation."""
        if not self._scanning:
            return
        
        # In a real system, this would track real scanner hardware position
        # Here we just visualize that something is happening
        # This position is not used for the actual data generation
        
        settings = self._scan_settings
        if settings is None:
            return
        
        # Simulate scanner position feedback
        for scan_pos in self.module_state.sigStateChanged.emit():
            pass  # This would emit position updates

    def _create_random_nv_centers(self):
        """Create random NV centers in the scan area."""
        self._nv_centers = []
        
        # Calculate area in μm²
        area = 1.0
        for axis, axis_obj in self._scanner_axes.items():
            min_val, max_val = axis_obj.position.bounds
            area *= (max_val - min_val)
        
        # Calculate number of NV centers based on density and area
        num_centers = max(1, int(round(area * self._nv_density)))
        
        # Generate random NV centers
        for i in range(num_centers):
            # Random position within scanning range
            position = {}
            for axis, axis_obj in self._scanner_axes.items():
                min_val, max_val = axis_obj.position.bounds
                position[axis] = min_val + random.random() * (max_val - min_val)
            
            # Random properties
            nv_center = {
                'id': f"NV{i+1}",
                'position': position,
                'brightness': 0.5 + random.random() * 0.5,  # Random brightness between 0.5 and 1.0
                'spin': {
                    'D': 2.87e9 + random.gauss(0, 1e6),  # Zero-field splitting with some variation
                    'E': random.gauss(0, 2e6),  # Strain splitting
                }
            }
            
            self._nv_centers.append(nv_center)
            
        self.log.info(f"Created {len(self._nv_centers)} simulated NV centers")
        
    def get_nv_center_at_position(self, position):
        """Find the closest NV center to a given position."""
        closest_nv = None
        min_distance = float('inf')
        
        for nv in self._nv_centers:
            # Calculate squared distance to NV center
            distance_sq = 0
            for axis, pos in position.items():
                if axis in nv['position']:
                    distance_sq += (pos - nv['position'][axis])**2
            
            # Convert to distance in μm
            distance = np.sqrt(distance_sq)
            
            if distance < min_distance:
                min_distance = distance
                closest_nv = nv
        
        # Only return if the NV center is close enough (within the PSF)
        if min_distance < 0.5:  # 500 nm
            return closest_nv
            
        return None