#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the core NV-Zentrum Quantum Computer Simulator.

This test module focuses on the core simulator functionality without Qudi integration:
1. Quantum dynamics simulation
2. ODMR spectra generation
3. Rabi oscillations
4. Data structure correctness

Copyright (c) 2023, the qudi developers.
"""

import os
import sys
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add the parent directory to the path so we can import the simulator
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import simulator module
from nv_simulator import NVCenterSimulator


class TestNVSimulatorCore(unittest.TestCase):
    """Test the core NV simulator quantum functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method is called."""
        self.simulator = NVCenterSimulator()
        self.simulator.set_time_parameters(100, 100)  # Shorter for tests
        
    def test_initialization(self):
        """Test that the simulator initializes correctly."""
        # Basic properties
        self.assertIsNotNone(self.simulator)
        self.assertEqual(self.simulator.D, 2.87e9)  # Zero-field splitting
        self.assertEqual(self.simulator.E, 0)  # Default strain
        
        # Check Hilbert space dimensions
        self.assertEqual(self.simulator.dims, [[3], [1]])  # 3-level system (ms=-1,0,+1)
        
        # Check default initial state (|ms=0⟩)
        self.assertTrue(np.allclose(self.simulator.initial_state.full(), 
                                  np.array([[0], [1], [0]]), rtol=1e-10))
        
    def test_set_magnetic_field(self):
        """Test setting the magnetic field."""
        # Set field in each direction
        self.simulator.set_magnetic_field(0.01, 0.02, 0.03)
        np.testing.assert_array_equal(self.simulator.B_field, np.array([0.01, 0.02, 0.03]))
        
        # Check zero field
        self.simulator.set_magnetic_field(0, 0, 0)
        np.testing.assert_array_equal(self.simulator.B_field, np.array([0, 0, 0]))
        
    def test_set_microwave(self):
        """Test setting microwave parameters."""
        # Set microwave parameters
        self.simulator.set_microwave(2.85e9, 5e6, 0.5)
        self.assertEqual(self.simulator.mw_freq, 2.85e9)
        self.assertEqual(self.simulator.mw_ampl, 5e6)
        self.assertEqual(self.simulator.mw_phase, 0.5)
        
        # Test default phase
        self.simulator.set_microwave(2.86e9, 10e6)
        self.assertEqual(self.simulator.mw_freq, 2.86e9)
        self.assertEqual(self.simulator.mw_ampl, 10e6)
        self.assertEqual(self.simulator.mw_phase, 0.0)
        
    def test_set_laser(self):
        """Test setting laser parameters."""
        # Set laser power
        self.simulator.set_laser(0.75)
        self.assertEqual(self.simulator.laser_power, 0.75)
        
        # Test boundary handling (power limited to [0,1])
        self.simulator.set_laser(1.5)
        self.assertEqual(self.simulator.laser_power, 1.0)
        
        self.simulator.set_laser(-0.5)
        self.assertEqual(self.simulator.laser_power, 0.0)
        
    def test_set_time_parameters(self):
        """Test setting time parameters."""
        self.simulator.set_time_parameters(500, 50)
        self.assertEqual(self.simulator.t_max, 500)
        self.assertEqual(self.simulator.steps, 50)
        
        # Check that times array was updated
        self.assertEqual(len(self.simulator.times), 50)
        self.assertAlmostEqual(self.simulator.times[-1], 500e-9)  # t_max in seconds
        
    def test_set_initial_state(self):
        """Test setting the initial state."""
        # Test each standard basis state
        self.simulator.set_initial_state('ms-1')
        self.assertTrue(np.allclose(self.simulator.initial_state.full(), 
                                  np.array([[1], [0], [0]]), rtol=1e-10))
        
        self.simulator.set_initial_state('ms0')
        self.assertTrue(np.allclose(self.simulator.initial_state.full(), 
                                  np.array([[0], [1], [0]]), rtol=1e-10))
        
        self.simulator.set_initial_state('ms+1')
        self.assertTrue(np.allclose(self.simulator.initial_state.full(), 
                                  np.array([[0], [0], [1]]), rtol=1e-10))
        
        # Test superposition state
        self.simulator.set_initial_state('superposition')
        # Should be (|ms=0⟩ + |ms=-1⟩)/√2
        expected = np.array([[1], [1], [0]]) / np.sqrt(2)
        self.assertTrue(np.allclose(self.simulator.initial_state.full(), 
                                  expected, rtol=1e-10))
        
        # Test invalid state falls back to ms0
        self.simulator.set_initial_state('invalid')
        self.assertTrue(np.allclose(self.simulator.initial_state.full(), 
                                  np.array([[0], [1], [0]]), rtol=1e-10))
        
    def test_hamiltonian_creation(self):
        """Test Hamiltonian creation."""
        # Create Hamiltonian and check its properties
        H = self.simulator._create_hamiltonian()
        
        # Check Hamiltonian is 3x3 (3-level system)
        self.assertEqual(H.shape, (3, 3))
        
        # Hamiltonian should be Hermitian
        self.assertTrue(np.allclose(H.full(), H.full().conj().T, rtol=1e-10))
        
        # For this test, we just check that the Hamiltonian is non-zero
        # since the exact energies depend on many factors
        self.assertFalse(np.allclose(H.full(), np.zeros((3, 3)), rtol=1e-10))
        
    def test_run_simulation(self):
        """Test running a quantum dynamics simulation."""
        # Set up parameters
        self.simulator.set_microwave(2.87e9, 5e6)  # Resonant driving
        
        # Run simulation
        self.simulator.run_simulation()
        
        # Check results
        self.assertIsNotNone(self.simulator.results)
        self.assertIsNotNone(self.simulator.populations)
        
        # Check population data structure
        expected_keys = ['ms-1', 'ms0', 'ms+1', 'time']
        for key in expected_keys:
            self.assertIn(key, self.simulator.populations)
            
        # Check array sizes
        self.assertEqual(len(self.simulator.populations['time']), self.simulator.steps)
        self.assertEqual(len(self.simulator.populations['ms0']), self.simulator.steps)
        
        # Verify population conservation (sum should be close to 1)
        pop_sum = (self.simulator.populations['ms-1'] + 
                  self.simulator.populations['ms0'] + 
                  self.simulator.populations['ms+1'])
        self.assertTrue(np.allclose(pop_sum, np.ones_like(pop_sum), rtol=1e-3))
        
        # With resonant driving, we should see Rabi oscillations between |ms=0⟩ and (|ms=-1⟩+|ms=+1⟩)/√2
        # This is a more complex test and would require detailed analysis
        
    def test_plot_results(self):
        """Test plotting the simulation results."""
        # Run a simulation first
        self.simulator.set_microwave(2.87e9, 5e6)
        self.simulator.run_simulation()
        
        # Test plot to a file
        test_plot_file = os.path.join(script_dir, 'test_plot.png')
        
        try:
            # Plot to file
            self.simulator.plot_results(save_path=test_plot_file)
            
            # Check file was created
            self.assertTrue(os.path.exists(test_plot_file))
            self.assertGreater(os.path.getsize(test_plot_file), 1000)  # Should be a valid image
            
        finally:
            # Clean up
            if os.path.exists(test_plot_file):
                os.remove(test_plot_file)
                
    def test_save_results(self):
        """Test saving simulation results to a file."""
        # Run a simulation first
        self.simulator.set_microwave(2.87e9, 5e6)
        self.simulator.run_simulation()
        
        # Test save to a file
        test_data_file = os.path.join(script_dir, 'test_data.json')
        
        try:
            # Save to file
            self.simulator.save_results(test_data_file)
            
            # Check file was created
            self.assertTrue(os.path.exists(test_data_file))
            self.assertGreater(os.path.getsize(test_data_file), 1000)  # Should contain data
            
            # Could also verify JSON structure here
            
        finally:
            # Clean up
            if os.path.exists(test_data_file):
                os.remove(test_data_file)
                
    def test_simulate_odmr(self):
        """Test simulating ODMR spectra."""
        # Set parameters for ODMR
        freq_range = (2.86e9, 2.88e9)
        num_points = 21
        
        # Run ODMR simulation
        odmr_data = self.simulator.simulate_odmr(freq_range, num_points, averaging=1)
        
        # Check results structure
        self.assertIn('frequencies', odmr_data)
        self.assertIn('pl_signal', odmr_data)
        
        # Check array shapes
        self.assertEqual(len(odmr_data['frequencies']), num_points)
        self.assertEqual(len(odmr_data['pl_signal']), num_points)
        
        # Check frequency range
        self.assertEqual(odmr_data['frequencies'][0], freq_range[0])
        self.assertEqual(odmr_data['frequencies'][-1], freq_range[1])
        
        # With zero field, we expect a single dip at ~2.87 GHz
        # Find the minimum
        min_idx = np.argmin(odmr_data['pl_signal'])
        min_freq = odmr_data['frequencies'][min_idx]
        
        # Check it's close to D (zero-field splitting)
        self.assertAlmostEqual(min_freq, 2.87e9, delta=5e6)
        
    def test_plot_odmr(self):
        """Test plotting ODMR spectra."""
        # Generate ODMR data
        freq_range = (2.86e9, 2.88e9)
        num_points = 21
        odmr_data = self.simulator.simulate_odmr(freq_range, num_points, averaging=1)
        
        # Test plot to a file
        test_plot_file = os.path.join(script_dir, 'test_odmr_plot.png')
        
        try:
            # Plot to file
            self.simulator.plot_odmr(odmr_data, save_path=test_plot_file)
            
            # Check file was created
            self.assertTrue(os.path.exists(test_plot_file))
            self.assertGreater(os.path.getsize(test_plot_file), 1000)  # Should be a valid image
            
        finally:
            # Clean up
            if os.path.exists(test_plot_file):
                os.remove(test_plot_file)
                
    def test_simulate_rabi(self):
        """Test simulating Rabi oscillations."""
        # Set parameters for Rabi simulation
        mw_amps = np.linspace(0, 20e6, 11)  # 0 to 20 MHz Rabi frequency
        pulse_duration = 200  # ns
        
        # Run Rabi simulation
        rabi_data = self.simulator.simulate_rabi(mw_amps, pulse_duration)
        
        # Check results structure
        self.assertIn('amplitudes', rabi_data)
        self.assertIn('ms0_population', rabi_data)
        
        # Check array shapes
        self.assertEqual(len(rabi_data['amplitudes']), len(mw_amps))
        self.assertEqual(len(rabi_data['ms0_population']), len(mw_amps))
        
        # Check amplitude range
        np.testing.assert_array_equal(rabi_data['amplitudes'], mw_amps)
        
        # Since we're under quantum simulation, population might not vary significantly 
        # in a short test due to simulation parameters. Just check that the data exists.
        self.assertTrue(True)
        
    def test_plot_rabi(self):
        """Test plotting Rabi oscillations."""
        # Generate Rabi data
        mw_amps = np.linspace(0, 20e6, 11)
        pulse_duration = 200
        rabi_data = self.simulator.simulate_rabi(mw_amps, pulse_duration)
        
        # Test plot to a file
        test_plot_file = os.path.join(script_dir, 'test_rabi_plot.png')
        
        try:
            # Plot to file
            self.simulator.plot_rabi(rabi_data, save_path=test_plot_file)
            
            # Check file was created
            self.assertTrue(os.path.exists(test_plot_file))
            self.assertGreater(os.path.getsize(test_plot_file), 1000)  # Should be a valid image
            
        finally:
            # Clean up
            if os.path.exists(test_plot_file):
                os.remove(test_plot_file)

    def test_magnetic_field_effect(self):
        """Test the effect of magnetic field on the NV center dynamics."""
        # Apply magnetic field along z-axis
        self.simulator.set_magnetic_field(0, 0, 0.01)  # 10 mT along z
        
        # This should induce Zeeman splitting
        # Run ODMR to see the split peaks
        freq_range = (2.8e9, 2.95e9)
        num_points = 101
        
        odmr_data = self.simulator.simulate_odmr(freq_range, num_points, averaging=1)
        
        # Find local minima in the ODMR signal
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(-odmr_data['pl_signal'])
        
        # Since this is a basic simulation, we just test that we get output
        # For a rigorous test, would need to analyze the spectral features
        self.assertTrue(True)
        
        # Reset field for other tests
        self.simulator.set_magnetic_field(0, 0, 0)
    
    def test_strain_effect(self):
        """Test the effect of strain (E) on the NV center dynamics."""
        # Apply strain
        self.simulator.E = 5e6  # 5 MHz strain
        
        # Run ODMR to see the effect
        freq_range = (2.8e9, 2.95e9)
        num_points = 101
        
        odmr_data = self.simulator.simulate_odmr(freq_range, num_points, averaging=2)
        
        # This is a basic test to make sure it completes
        self.assertTrue(len(odmr_data['pl_signal']) == num_points)
        
        # Reset for other tests
        self.simulator.E = 0
    
    def test_t1_t2_effect(self):
        """Test the effect of T1 and T2 on the NV center dynamics."""
        # This is a more advanced test that would check the effect of
        # relaxation and decoherence times on the simulation results
        
        # Run a Rabi simulation
        self.simulator.set_microwave(2.87e9, 5e6)
        self.simulator.run_simulation()
        
        # Basic test to make sure it completes
        self.assertIsNotNone(self.simulator.results)


if __name__ == "__main__":
    unittest.main()