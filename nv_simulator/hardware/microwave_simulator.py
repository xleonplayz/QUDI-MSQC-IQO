# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware file to control the simulated microwave source for NV centers.

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

import time
import numpy as np
import os
import sys
import threading

# Add the nv_simulator directory to the path so we can import the simulator
script_dir = os.path.dirname(os.path.abspath(__file__))
simulator_dir = os.path.dirname(script_dir)
if simulator_dir not in sys.path:
    sys.path.append(simulator_dir)

from nv_simulator import NVCenterSimulator

from qudi.interface.microwave_interface import MicrowaveInterface, MicrowaveConstraints
from qudi.util.enums import SamplingOutputMode
from qudi.util.mutex import Mutex


class MicrowaveSimulator(MicrowaveInterface):
    """A qudi hardware module to simulate a microwave source for NV center experiments.

    Example config for copy-paste:

    mw_source_simulator:
        module.Class: 'nv_simulator.hardware.microwave_simulator.MicrowaveSimulator'
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = Mutex()
        self._constraints = None

        self._cw_power = 0.
        self._cw_frequency = 2.87e9
        self._scan_power = 0.
        self._scan_frequencies = None
        self._scan_sample_rate = -1.
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._is_scanning = False
        
        # NV center simulator instance
        self._nv_simulator = None
        self._simulation_thread = None
        self._stop_simulation = False

    def on_activate(self):
        """Initialisation performed during activation of the module."""
        self._constraints = MicrowaveConstraints(
            power_limits=(-60.0, 30),
            frequency_limits=(100e3, 20e9),
            scan_size_limits=(2, 1001),
            sample_rate_limits=(0.1, 200),
            scan_modes=(SamplingOutputMode.JUMP_LIST, SamplingOutputMode.EQUIDISTANT_SWEEP)
        )

        self._cw_power = self._constraints.min_power + (
                    self._constraints.max_power - self._constraints.min_power) / 2
        self._cw_frequency = 2.87e9
        self._scan_power = self._cw_power
        self._scan_frequencies = None
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._scan_sample_rate = 100
        self._is_scanning = False
        
        # Initialize the NV center simulator
        self._nv_simulator = NVCenterSimulator()
        self._nv_simulator.set_time_parameters(1000, 1000)  # Default simulation time parameters
        self._nv_simulator.set_initial_state('ms0')
        self._stop_simulation = False

    def on_deactivate(self):
        """Cleanup performed during deactivation of the module."""
        # Stop any running simulation
        self._stop_simulation = True
        if self._simulation_thread is not None and self._simulation_thread.is_alive():
            self._simulation_thread.join(timeout=5)
        
        self._nv_simulator = None

    @property
    def constraints(self):
        """The microwave constraints object for this device.

        @return MicrowaveConstraints:
        """
        return self._constraints

    @property
    def is_scanning(self):
        """Read-Only boolean flag indicating if a scan is running at the moment. Can be used
        together with module_state() to determine if the currently running microwave output is a
        scan or CW.
        Should return False if module_state() is 'idle'.

        @return bool: Flag indicating if a scan is running (True) or not (False)
        """
        with self._thread_lock:
            return self._is_scanning

    @property
    def cw_power(self):
        """Read-only property returning the currently configured CW microwave power in dBm.

        @return float: The currently set CW microwave power in dBm.
        """
        with self._thread_lock:
            return self._cw_power

    @property
    def cw_frequency(self):
        """Read-only property returning the currently set CW microwave frequency in Hz.

        @return float: The currently set CW microwave frequency in Hz.
        """
        with self._thread_lock:
            return self._cw_frequency

    @property
    def scan_power(self):
        """Read-only property returning the currently configured microwave power in dBm used for
        scanning.

        @return float: The currently set scanning microwave power in dBm
        """
        with self._thread_lock:
            return self._scan_power

    @property
    def scan_frequencies(self):
        """Read-only property returning the currently configured microwave frequencies used for
        scanning.

        In case of self.scan_mode == SamplingOutputMode.JUMP_LIST, this will be a 1D numpy array.
        In case of self.scan_mode == SamplingOutputMode.EQUIDISTANT_SWEEP, this will be a tuple
        containing 3 values (freq_begin, freq_end, number_of_samples).
        If no frequency scan has been configured, return None.

        @return float[]: The currently set scanning frequencies. None if not set.
        """
        with self._thread_lock:
            return self._scan_frequencies

    @property
    def scan_mode(self):
        """Read-only property returning the currently configured scan mode Enum.

        @return SamplingOutputMode: The currently set scan mode Enum
        """
        with self._thread_lock:
            return self._scan_mode

    @property
    def scan_sample_rate(self):
        """Read-only property returning the currently configured scan sample rate in Hz.

        @return float: The currently set scan sample rate in Hz
        """
        with self._thread_lock:
            return self._scan_sample_rate

    def off(self):
        """Switches off any microwave output (both scan and CW).
        Must return AFTER the device has actually stopped.
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.log.debug('Microwave output was not active')
                return
            self.log.debug('Stopping microwave output')
            
            # Stop any running simulation
            self._stop_simulation = True
            if self._simulation_thread is not None and self._simulation_thread.is_alive():
                self._simulation_thread.join(timeout=5)
            
            time.sleep(0.5)  # Simulate hardware delay
            self._is_scanning = False
            self.module_state.unlock()

    def set_cw(self, frequency, power):
        """Configure the CW microwave output. Does not start physical signal output, see also
        "cw_on".

        @param float frequency: frequency to set in Hz
        @param float power: power to set in dBm
        """
        with self._thread_lock:
            # Check if CW parameters can be set.
            if self.module_state() != 'idle':
                raise RuntimeError(
                    'Unable to set CW power and frequency. Microwave output is active.'
                )
            self._assert_cw_parameters_args(frequency, power)

            # Set power and frequency
            self.log.debug(f'Setting CW power to {power} dBm and frequency to {frequency:.9e} Hz')
            self._cw_power = power
            self._cw_frequency = frequency
            
            # Also set these values in the simulator
            self._nv_simulator.mw_freq = frequency
            self._nv_simulator.mw_ampl = self._convert_dbm_to_rabi_freq(power)

    def cw_on(self):
        """Switches on cw microwave output.

        Must return AFTER the output is actually active.
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.log.debug(f'Starting CW microwave output with {self._cw_frequency:.6e} Hz '
                               f'and {self._cw_power:.6f} dBm')
                
                # Configure the simulator for CW mode
                self._nv_simulator.set_microwave(self._cw_frequency, 
                                               self._convert_dbm_to_rabi_freq(self._cw_power))
                
                # Start a simulation thread
                self._stop_simulation = False
                self._simulation_thread = threading.Thread(target=self._run_cw_simulation)
                self._simulation_thread.daemon = True
                self._simulation_thread.start()
                
                time.sleep(0.5)  # Simulate hardware initialization time
                self._is_scanning = False
                self.module_state.lock()
            elif self._is_scanning:
                raise RuntimeError(
                    'Unable to start microwave CW output. Frequency scanning in progress.'
                )
            else:
                self.log.debug('CW microwave output already running')

    def configure_scan(self, power, frequencies, mode, sample_rate):
        """Configure a frequency scan.

        @param float power: the power in dBm to be used during the scan
        @param float[] frequencies: an array of all frequencies (jump list)
                                  or a tuple of start, stop frequency and number of steps (equidistant sweep)
        @param SamplingOutputMode mode: enum stating the way how the frequencies are defined
        @param float sample_rate: external scan trigger rate
        """
        with self._thread_lock:
            # Sanity checking
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to configure scan. Microwave output is active.')
            self._assert_scan_configuration_args(power, frequencies, mode, sample_rate)

            # Actually change settings
            time.sleep(0.5)  # Simulate hardware delay
            if mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                self._scan_frequencies = tuple(frequencies)
            else:
                self._scan_frequencies = np.asarray(frequencies, dtype=np.float64)
            self._scan_power = power
            self._scan_mode = mode
            self._scan_sample_rate = sample_rate
            self.log.debug(
                f'Scan configured in mode "{mode.name}" with {sample_rate:.9e} Hz sample rate, '
                f'{power} dBm power and frequencies:\n{self._scan_frequencies}.'
            )

    def start_scan(self):
        """Switches on the microwave scanning.

        Must return AFTER the output is actually active (and can receive triggers for example).
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError(
                    'Unable to start microwave frequency scan. Microwave output is active.'
                )
            
            # Start a simulation thread for scanning
            self._stop_simulation = False
            self._simulation_thread = threading.Thread(target=self._run_scan_simulation)
            self._simulation_thread.daemon = True
            self._simulation_thread.start()
            
            self.module_state.lock()
            self._is_scanning = True
            time.sleep(0.5)  # Simulate hardware initialization time
            self.log.debug(f'Starting frequency scan in "{self._scan_mode.name}" mode')

    def reset_scan(self):
        """Reset currently running scan and return to start frequency.
        Does not need to stop and restart the microwave output if the device allows soft scan reset.
        """
        with self._thread_lock:
            if self._is_scanning:
                self.log.debug('Frequency scan soft reset')
                
                # Stop current simulation and restart
                self._stop_simulation = True
                if self._simulation_thread is not None and self._simulation_thread.is_alive():
                    self._simulation_thread.join(timeout=2)
                
                # Restart scan
                self._stop_simulation = False
                self._simulation_thread = threading.Thread(target=self._run_scan_simulation)
                self._simulation_thread.daemon = True
                self._simulation_thread.start()
                
                time.sleep(0.5)  # Simulate hardware delay

    def _convert_dbm_to_rabi_freq(self, power_dbm):
        """Convert power in dBm to a Rabi frequency in Hz for the simulator.
        
        This is a simplistic model - in reality, this would depend on many factors like
        the specific microwave antenna, NV center coupling, etc.
        
        @param float power_dbm: Microwave power in dBm
        @return float: Rabi frequency in Hz
        """
        # Simple conversion model:
        # 0 dBm ~= 10 MHz Rabi frequency
        # For each 6 dB increase, Rabi frequency doubles (sqrt of power relationship)
        base_power = 0  # dBm
        base_rabi = 10e6  # Hz
        
        # Calculate ratio in linear scale
        power_ratio = 10**((power_dbm - base_power) / 20)  # 20 because amplitude ~ sqrt(power)
        return base_rabi * power_ratio

    def _run_cw_simulation(self):
        """Run a continuous wave simulation in a separate thread."""
        self.log.debug('Starting CW simulation thread')
        
        while not self._stop_simulation and self.module_state() != 'idle':
            # In a real system, this would interact with the physical NV center
            # Here we just periodically update the simulator state
            try:
                # Run a timestep of the simulation
                self._nv_simulator.run_simulation()
                
                # In a real implementation, you might want to:
                # 1. Use the results to generate events for other parts of the system
                # 2. Adjust parameters based on feedback
                # 3. Log important state changes
                
                time.sleep(0.1)  # Don't hog the CPU
            except Exception as e:
                self.log.error(f'Error in CW simulation: {str(e)}')
                break
        
        self.log.debug('CW simulation thread finished')

    def _run_scan_simulation(self):
        """Run a frequency scan simulation in a separate thread."""
        self.log.debug('Starting scan simulation thread')
        
        try:
            # Extract frequency range
            if self._scan_mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                freq_min, freq_max, num_points = self._scan_frequencies
                frequencies = np.linspace(freq_min, freq_max, int(num_points))
            else:
                frequencies = self._scan_frequencies
            
            # Convert power to Rabi frequency
            rabi_ampl = self._convert_dbm_to_rabi_freq(self._scan_power)
            
            # Set up ODMR simulation
            self._nv_simulator.mw_ampl = rabi_ampl
            
            # Run the simulation
            odmr_data = self._nv_simulator.simulate_odmr(
                freq_range=(np.min(frequencies), np.max(frequencies)),
                num_points=len(frequencies),
                averaging=1  # Keep this low for performance
            )
            
            # In a real system, you would be communicating with hardware
            # and collecting data as the frequencies are stepped through
            while not self._stop_simulation and self.module_state() != 'idle':
                time.sleep(0.1)  # Just simulate activity for now
            
        except Exception as e:
            self.log.error(f'Error in scan simulation: {str(e)}')
        
        self.log.debug('Scan simulation thread finished')