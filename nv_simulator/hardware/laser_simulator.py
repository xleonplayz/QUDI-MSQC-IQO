# -*- coding: utf-8 -*-

"""
This module acts like a laser for NV center experiments.

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

import math
import time
import random
import threading
import os
import sys
import numpy as np

# Add the nv_simulator directory to the path so we can import the simulator
script_dir = os.path.dirname(os.path.abspath(__file__))
simulator_dir = os.path.dirname(script_dir)
if simulator_dir not in sys.path:
    sys.path.append(simulator_dir)

from nv_simulator import NVCenterSimulator

from qudi.interface.simple_laser_interface import SimpleLaserInterface
from qudi.interface.simple_laser_interface import LaserState, ShutterState, ControlMode


class LaserSimulator(SimpleLaserInterface):
    """Laser simulator for NV center experiments.

    Example config for copy-paste:

    laser_simulator:
        module.Class: 'nv_simulator.hardware.laser_simulator.LaserSimulator'
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstate = LaserState.OFF
        self.shutter = ShutterState.CLOSED
        self.mode = ControlMode.POWER
        self.current_setpoint = 0
        self.power_setpoint = 0
        
        # NV center simulator instance
        self._nv_simulator = None
        self._simulation_thread = None
        self._stop_simulation = False
        
        # Simulated temperatures
        self._psu_temp = 30.0
        self._head_temp = 40.0

    def on_activate(self):
        """Activate module."""
        # Initialize the NV center simulator
        self._nv_simulator = NVCenterSimulator()
        self._nv_simulator.set_time_parameters(1000, 1000)  # Default simulation time parameters
        self._nv_simulator.set_initial_state('ms0')
        self._stop_simulation = False
        
        # Start a simulation thread for temperature evolution
        self._simulation_thread = threading.Thread(target=self._simulate_temperature)
        self._simulation_thread.daemon = True
        self._simulation_thread.start()

    def on_deactivate(self):
        """Deactivate module."""
        # Stop any running simulation
        self._stop_simulation = True
        if self._simulation_thread is not None and self._simulation_thread.is_alive():
            self._simulation_thread.join(timeout=5)
        
        self._nv_simulator = None

    def get_power_range(self):
        """Return optical power range

        @return float[2]: power range (min, max)
        """
        return 0, 0.250

    def get_power(self):
        """Return laser power

        @return float: Laser power in watts
        """
        # Add some noise to the power reading
        return self.power_setpoint * random.gauss(1, 0.01)

    def get_power_setpoint(self):
        """Return optical power setpoint.

        @return float: power setpoint in watts
        """
        return self.power_setpoint

    def set_power(self, power):
        """Set power setpoint.

        @param float power: power to set
        """
        self.power_setpoint = power
        self.current_setpoint = math.sqrt(4*self.power_setpoint)*100
        
        # Update the simulator laser power
        if self._nv_simulator is not None and self.lstate == LaserState.ON:
            self._nv_simulator.set_laser(min(1.0, power / 0.25))  # Normalize to [0, 1]

    def get_current_unit(self):
        """Get unit for laser current.

        @return str: unit
        """
        return '%'

    def get_current_range(self):
        """Get laser current range.

        @return float[2]: laser current range
        """
        return 0, 100

    def get_current(self):
        """Get actual laser current

        @return float: laser current in current units
        """
        # Add some noise to the current reading
        return self.current_setpoint * random.gauss(1, 0.05)

    def get_current_setpoint(self):
        """Get laser current setpoint

        @return float: laser current setpoint
        """
        return self.current_setpoint

    def set_current(self, current):
        """Set laser current setpoint

        @param float current: desired laser current setpoint
        """
        self.current_setpoint = current
        self.power_setpoint = math.pow(self.current_setpoint/100, 2) / 4
        
        # Update the simulator laser power
        if self._nv_simulator is not None and self.lstate == LaserState.ON:
            self._nv_simulator.set_laser(min(1.0, self.power_setpoint / 0.25))  # Normalize to [0, 1]

    def allowed_control_modes(self):
        """Get supported control modes

        @return frozenset: set of supported ControlMode enums
        """
        return frozenset({ControlMode.POWER, ControlMode.CURRENT})

    def get_control_mode(self):
        """Get the currently active control mode

        @return ControlMode: active control mode enum
        """
        return self.mode

    def set_control_mode(self, control_mode):
        """Set the active control mode

        @param ControlMode control_mode: desired control mode enum
        """
        self.mode = control_mode

    def on(self):
        """Turn on laser.

        @return LaserState: actual laser state
        """
        self.log.debug('Turning laser on')
        time.sleep(0.5)  # Simulate hardware delay
        self.lstate = LaserState.ON
        
        # Update the simulator
        if self._nv_simulator is not None:
            self._nv_simulator.set_laser(min(1.0, self.power_setpoint / 0.25))  # Normalize to [0, 1]
        
        return self.lstate

    def off(self):
        """Turn off laser.

        @return LaserState: actual laser state
        """
        self.log.debug('Turning laser off')
        time.sleep(0.5)  # Simulate hardware delay
        self.lstate = LaserState.OFF
        
        # Update the simulator
        if self._nv_simulator is not None:
            self._nv_simulator.set_laser(0.0)
        
        return self.lstate

    def get_laser_state(self):
        """Get laser state

        @return LaserState: current laser state
        """
        return self.lstate

    def set_laser_state(self, state):
        """Set laser state.

        @param LaserState state: desired laser state enum
        """
        self.log.debug(f'Setting laser state to {state}')
        time.sleep(0.5)  # Simulate hardware delay
        self.lstate = state
        
        # Update the simulator
        if self._nv_simulator is not None:
            if state == LaserState.ON:
                self._nv_simulator.set_laser(min(1.0, self.power_setpoint / 0.25))  # Normalize to [0, 1]
            else:
                self._nv_simulator.set_laser(0.0)
        
        return self.lstate

    def get_shutter_state(self):
        """Get laser shutter state

        @return ShutterState: actual laser shutter state
        """
        return self.shutter

    def set_shutter_state(self, state):
        """Set laser shutter state.

        @param ShutterState state: desired laser shutter state
        """
        self.log.debug(f'Setting shutter state to {state}')
        time.sleep(0.5)  # Simulate hardware delay
        self.shutter = state
        
        # Update the simulator based on shutter state
        if self._nv_simulator is not None and self.lstate == LaserState.ON:
            if state == ShutterState.OPEN:
                self._nv_simulator.set_laser(min(1.0, self.power_setpoint / 0.25))  # Normalize to [0, 1]
            else:
                self._nv_simulator.set_laser(0.0)
        
        return self.shutter

    def get_temperatures(self):
        """Get all available temperatures.

        @return dict: dict of temperature names and value in degrees Celsius
        """
        return {
            'psu': self._psu_temp * random.gauss(1, 0.01),
            'head': self._head_temp * random.gauss(1, 0.02)
        }

    def get_extra_info(self):
        """Multiple lines of diagnostic information

        @return str: Simulator info
        """
        return "NV Center Laser Simulator\nWavelength: 532 nm\nMax Power: 250 mW\nSimulated Temperature Control"

    def _simulate_temperature(self):
        """Simulate temperature evolution of the laser system."""
        self.log.debug('Starting temperature simulation thread')
        
        while not self._stop_simulation:
            try:
                # Temperatures slowly drift and depend on laser state
                if self.lstate == LaserState.ON:
                    # When laser is on, temperatures slowly rise to a higher equilibrium
                    self._psu_temp += 0.01 * (35.0 - self._psu_temp) + 0.1 * random.gauss(0, 0.01)
                    self._head_temp += 0.01 * (45.0 - self._head_temp) + 0.1 * random.gauss(0, 0.01)
                else:
                    # When laser is off, temperatures drift toward room temperature
                    self._psu_temp += 0.01 * (30.0 - self._psu_temp) + 0.1 * random.gauss(0, 0.01)
                    self._head_temp += 0.01 * (40.0 - self._head_temp) + 0.1 * random.gauss(0, 0.01)
                
                # Limit to reasonable ranges
                self._psu_temp = max(20.0, min(50.0, self._psu_temp))
                self._head_temp = max(25.0, min(60.0, self._head_temp))
                
                time.sleep(0.2)  # Update temperature at 5 Hz
            except Exception as e:
                self.log.error(f'Error in temperature simulation: {str(e)}')
                time.sleep(1.0)  # Wait a bit before trying again
        
        self.log.debug('Temperature simulation thread finished')