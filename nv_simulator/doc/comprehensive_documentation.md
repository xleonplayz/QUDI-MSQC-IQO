# NV-Center Quantum Computer Simulator: Comprehensive Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Simulator Implementation](#core-simulator-implementation)
4. [Hardware Interface Modules](#hardware-interface-modules)
5. [Integration with Qudi Framework](#integration-with-qudi-framework)
6. [Quantum Physical Model](#quantum-physical-model)
7. [Testing Framework](#testing-framework)
8. [Usage Examples](#usage-examples)
9. [API Reference](#api-reference)
10. [Future Development](#future-development)

## Introduction

The NV-Center Quantum Computer Simulator is a comprehensive software package designed to simulate the quantum dynamics of Nitrogen-Vacancy (NV) centers in diamond. This simulator serves dual purposes:

1. **Standalone Operation**: Can be used independently for quantum research and education
2. **Qudi Integration**: Seamlessly integrates with the Qudi laboratory framework

The simulator accurately models the quantum behavior of NV centers, including:
- Zero-field splitting (D ≈ 2.87 GHz)
- Strain effects (E parameter)
- Magnetic field interactions (Zeeman effect)
- Microwave-driven spin dynamics (Rabi oscillations)
- Optical initialization and readout
- Spatial dependence for confocal imaging

This documentation provides a comprehensive overview of the simulator's architecture, implementation, physical model, and integration capabilities.

## Architecture Overview

The simulator is structured as a modular package with the following components:

```
nv_simulator/
├── __init__.py                  # Package initialization
├── nv_simulator.py              # Core quantum simulator
├── requirements.txt             # Dependencies
├── examples.sh                  # Example usage scripts
├── example_config.yml           # Example Qudi configuration
├── hardware/                    # Hardware interface implementations
│   ├── __init__.py              # Hardware package initialization
│   ├── microwave_simulator.py   # Microwave interface
│   ├── laser_simulator.py       # Laser interface
│   └── scanning_probe_simulator.py # Scanning probe interface
├── test/                        # Testing framework
│   ├── __init__.py              # Test package initialization
│   ├── test_simulator_core.py   # Unit tests for core simulator
│   ├── test_simulator_integration.py # Integration tests with Qudi
│   └── run_tests.py             # Test runner
└── doc/                         # Documentation
    ├── usage_guide.md           # Usage documentation
    ├── api_reference.md         # API reference
    └── comprehensive_documentation.md # This document
```

The architecture follows a layered approach:
1. **Core Layer**: Quantum physics simulation using QuTiP
2. **Interface Layer**: Implementation of Qudi hardware interfaces
3. **Integration Layer**: Connection to Qudi modules and GUI

## Core Simulator Implementation

The `nv_simulator.py` module implements the `NVCenterSimulator` class, which is the central component handling quantum dynamics simulation.

### Key Features

- **Quantum State Management**: Tracks the quantum state of the NV center using density matrix formalism
- **Hamiltonian Construction**: Dynamically builds system Hamiltonians based on physical parameters
- **Time Evolution**: Solves the Lindblad master equation for open quantum system dynamics
- **Parameter Control**: Provides interfaces to adjust physical parameters like magnetic field, strain, temperature
- **Measurement Functions**: Simulates various measurement protocols (ODMR, Rabi, Hahn-Echo, etc.)

### Implementation Details

The core simulator uses QuTiP's master equation solver to simulate the time evolution of the NV center's quantum state. The Hamiltonian includes:

1. **Zero-Field Splitting**: $$H_{ZFS} = D S_z^2 + E(S_x^2 - S_y^2)$$
2. **Zeeman Interaction**: $$H_{Zeeman} = g_e \mu_B \vec{B} \cdot \vec{S}$$
3. **Microwave Drive**: $$H_{MW} = \Omega \cos(\omega t + \phi) S_x$$

Dissipation effects are modeled using collapse operators for:
- Longitudinal relaxation (T₁)
- Transverse relaxation (T₂)
- Optical pumping (when laser is on)

The class provides method interfaces for all typical NV center experiments:
- `run_odmr_simulation()`
- `run_rabi_simulation()`
- `run_ramsey_simulation()`
- `run_hahn_echo_simulation()`
- `simulate_confocal_scan()`

## Hardware Interface Modules

The hardware directory contains implementations of Qudi hardware interfaces using the core simulator.

### Microwave Simulator

The `microwave_simulator.py` module implements the `MicrowaveSimulator` class, which conforms to Qudi's `MicrowaveInterface`.

Key features:
- Simulates control of microwave frequency, power, and phase
- Handles CW and pulsed operation modes
- Properly triggers quantum state updates in the core simulator
- Includes realistic behavior like sweep times and frequency-dependent power

### Laser Simulator

The `laser_simulator.py` module implements the `LaserSimulator` class, which conforms to Qudi's `SimpleLaserInterface`.

Key features:
- Controls laser power and operating state
- Models wavelength-specific interactions with NV centers
- Simulates realistic power stabilization dynamics
- Triggers optical pumping in the quantum simulator when activated

### Scanning Probe Simulator

The `scanning_probe_simulator.py` module implements the `ScanningProbeSimulator` class, which conforms to Qudi's `ScanningProbeInterface`.

Key features:
- Simulates 3D confocal microscope positioning
- Models point spread function (PSF) for realistic imaging
- Incorporates realistic motor movement times
- Provides simulated fluorescence readout dependent on NV center state and laser power
- Simulates multiple NV centers at different 3D positions with varying properties

## Integration with Qudi Framework

The simulator is designed to integrate seamlessly with the Qudi laboratory framework, replacing dummy hardware modules with physically realistic simulations while maintaining API compatibility.

### Configuration Integration

The `example_config.yml` file demonstrates how to configure Qudi to use the simulator modules:

```yaml
hardware:
    microwave_simulator:
        module.Class: 'nv_simulator.hardware.microwave_simulator.MicrowaveSimulator'
        connect:
            nv_simulator: 'nv_simulator'
            
    laser_simulator:
        module.Class: 'nv_simulator.hardware.laser_simulator.LaserSimulator'
        connect:
            nv_simulator: 'nv_simulator'
            
    scanner_simulator:
        module.Class: 'nv_simulator.hardware.scanning_probe_simulator.ScanningProbeSimulator'
        connect:
            nv_simulator: 'nv_simulator'
            
    nv_simulator:
        module.Class: 'nv_simulator.nv_simulator.NVCenterSimulator'
        options:
            d_zero_splitting: 2.87e9  # Hz
            e_strain: 5e6  # Hz
            temperature: 300  # K
            t1_time: 1e-3  # s
            t2_time: 2e-6  # s
```

### Connector Architecture

The simulator uses Qudi's connector framework to establish links between hardware modules. The core simulator is instantiated first, and hardware interfaces connect to it using Qudi's connector mechanism.

Thread safety is maintained through Mutex locks for all operations that modify the quantum state, ensuring that simultaneous operations from different hardware modules don't cause race conditions.

## Quantum Physical Model

The simulator implements a detailed physical model of the NV center quantum system.

### Electronic Structure

The NV center is modeled as a spin-1 system with three levels:
- |0⟩: ms = 0 ground state
- |+1⟩: ms = +1 excited state  
- |-1⟩: ms = -1 excited state

The zero-field splitting separates the |0⟩ state from the |±1⟩ states by approximately 2.87 GHz at room temperature.

### Key Physical Parameters

- **D (Zero-Field Splitting)**: 2.87 GHz, temperature-dependent
- **E (Strain)**: Typically 1-10 MHz, causes splitting between |+1⟩ and |-1⟩ states
- **g-factor**: 2.0028 for electron spin
- **T₁ Time**: Longitudinal relaxation time, typically 1-6 ms at room temperature
- **T₂ Time**: Transverse coherence time, typically 1-2 μs without decoupling
- **T₂* Time**: Inhomogeneous dephasing time, typically 100-500 ns

### Optical Dynamics

The simulator models:
- Spin-dependent fluorescence (higher for |0⟩ state)
- Optical polarization into the |0⟩ state under green laser excitation
- Realistic fluorescence collection efficiency based on confocal PSF

### Environmental Effects

The model incorporates:
- Temperature dependence of D parameter
- Magnetic noise from nuclear spin bath
- Power fluctuations in microwave and laser sources
- Spatial dependence of the coupling strengths

## Testing Framework

The simulator includes a comprehensive testing framework to ensure accuracy and reliability.

### Unit Tests

The `test_simulator_core.py` file contains unit tests for the core simulator functionality:
- Hamiltonian construction tests
- Eigenvalue validation
- Time evolution verification
- Parameter boundary testing
- Conservation law checks

### Integration Tests

The `test_simulator_integration.py` file contains tests that verify correct integration with Qudi:
- Interface compliance tests
- End-to-end simulation tests
- Performance benchmarks
- Edge case handling

### Test Runner

The `run_tests.py` script provides a convenient way to run all tests with options for:
- Verbosity control
- Test filtering
- Coverage reporting
- Performance profiling

## Usage Examples

### Standalone Usage

```python
from nv_simulator import NVCenterSimulator

# Create simulator with custom parameters
simulator = NVCenterSimulator(
    d_zero_splitting=2.87e9,  # Hz
    e_strain=5e6,  # Hz
    temperature=300,  # K
    t1_time=1e-3,  # s
    t2_time=2e-6  # s
)

# Run an ODMR simulation
frequencies = np.linspace(2.8e9, 2.95e9, 100)
odmr_signal = simulator.run_odmr_simulation(
    frequencies=frequencies,
    magnetic_field=[0, 0, 10e-3]  # 10 mT along z
)

# Run a Rabi oscillation simulation
times = np.linspace(0, 1e-6, 100)
rabi_signal = simulator.run_rabi_simulation(
    times=times,
    rabi_frequency=10e6,  # Hz
    detuning=0  # Hz
)

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(frequencies / 1e9, odmr_signal)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Normalized Fluorescence')
plt.title('ODMR Spectrum')

plt.subplot(122)
plt.plot(times * 1e6, rabi_signal)
plt.xlabel('Time (μs)')
plt.ylabel('Population')
plt.title('Rabi Oscillations')
plt.tight_layout()
plt.show()
```

### Qudi Integration Usage

```python
# In a Qudi script or module

# Get the simulator modules from Qudi
simulator = self.nv_simulator()
microwave = self.microwave_simulator()
laser = self.laser_simulator()
scanner = self.scanner_simulator()

# Configure the microwave source
microwave.set_frequency(2.87e9)
microwave.set_power(-10)  # dBm
microwave.cw_on()

# Turn on the laser
laser.on()

# Perform a confocal scan
x_range = np.linspace(-5, 5, 100)  # μm
y_range = np.linspace(-5, 5, 100)  # μm
z_position = 0  # μm

image = np.zeros((len(x_range), len(y_range)))

for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        scanner.go_to_position((x, y, z_position))
        counts = scanner.get_scanner_count()
        image[i, j] = counts

# Visualize the confocal image
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(image, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]])
plt.colorbar(label='Counts')
plt.xlabel('X Position (μm)')
plt.ylabel('Y Position (μm)')
plt.title('Simulated Confocal Image')
plt.show()
```

## API Reference

### NVCenterSimulator Class

```python
class NVCenterSimulator:
    """Core simulator for NV center quantum dynamics."""
    
    def __init__(
        self, 
        d_zero_splitting=2.87e9,  # Hz
        e_strain=5e6,  # Hz
        temperature=300,  # K
        t1_time=1e-3,  # s
        t2_time=2e-6,  # s
        nv_positions=None,  # List of (x,y,z) positions in μm
        nv_orientations=None  # List of (theta,phi) orientations in radians
    ):
        """Initialize the NV center simulator with given parameters."""
        
    def set_magnetic_field(self, field_vector):
        """Set the magnetic field vector in Tesla [Bx, By, Bz]."""
        
    def set_microwave_driving(self, frequency, power, phase=0):
        """Set the microwave driving parameters."""
        
    def set_laser_power(self, power):
        """Set the laser power for optical pumping (0-100%)."""
        
    def run_odmr_simulation(self, frequencies, magnetic_field=None):
        """Run an ODMR spectrum simulation."""
        
    def run_rabi_simulation(self, times, rabi_frequency, detuning=0):
        """Run a Rabi oscillation simulation."""
        
    def run_ramsey_simulation(self, times, detuning):
        """Run a Ramsey free induction decay simulation."""
        
    def run_hahn_echo_simulation(self, times):
        """Run a Hahn echo simulation."""
        
    def simulate_confocal_scan(self, x_range, y_range, z_position):
        """Simulate a confocal microscope scan."""
        
    def get_current_state(self):
        """Return the current quantum state (density matrix)."""
        
    def get_hamiltonian(self):
        """Return the current system Hamiltonian."""
        
    def get_collapse_operators(self):
        """Return the current collapse operators."""
        
    def reset_state(self):
        """Reset the quantum state to thermal equilibrium."""
```

### MicrowaveSimulator Class

```python
class MicrowaveSimulator(MicrowaveInterface):
    """Simulated microwave source for NV center control."""
    
    def __init__(self):
        """Initialize the microwave simulator."""
        
    def on(self):
        """Turn the microwave source on."""
        
    def off(self):
        """Turn the microwave source off."""
        
    def get_power(self):
        """Return the current microwave power in dBm."""
        
    def set_power(self, power):
        """Set the microwave power in dBm."""
        
    def get_frequency(self):
        """Return the current frequency in Hz."""
        
    def set_frequency(self, frequency):
        """Set the microwave frequency in Hz."""
        
    def set_cw(self, frequency, power):
        """Set continuous wave mode with given frequency and power."""
        
    def list_scan(self, frequency_list, power):
        """Run a list scan over the given frequencies."""
        
    def frequency_sweep(self, start, stop, step, power):
        """Run a frequency sweep with the given parameters."""
        
    def set_ext_trigger(self, trigger_source, trigger_edge):
        """Set the external trigger source for pulsed operation."""
```

### LaserSimulator Class

```python
class LaserSimulator(SimpleLaserInterface):
    """Simulated laser source for NV center control."""
    
    def __init__(self):
        """Initialize the laser simulator."""
        
    def on(self):
        """Turn the laser on."""
        
    def off(self):
        """Turn the laser off."""
        
    def get_power(self):
        """Return the current laser power in percentage."""
        
    def set_power(self, power):
        """Set the laser power in percentage (0-100%)."""
        
    def get_power_range(self):
        """Return the laser power range."""
        
    def get_wavelength(self):
        """Return the laser wavelength in nm."""
        
    def get_wavelength_range(self):
        """Return the available wavelength range."""
        
    def set_wavelength(self, wavelength):
        """Set the laser wavelength in nm."""
        
    def get_extra_info(self):
        """Return extra information about the laser."""
```

### ScanningProbeSimulator Class

```python
class ScanningProbeSimulator(ScanningProbeInterface):
    """Simulated scanning probe for confocal microscopy."""
    
    def __init__(self):
        """Initialize the scanning probe simulator."""
        
    def reset(self):
        """Reset the scanning probe to home position."""
        
    def get_position(self):
        """Return the current position [x, y, z] in μm."""
        
    def get_position_range(self):
        """Return the available position range."""
        
    def set_position(self, position):
        """Set the position [x, y, z] in μm."""
        
    def get_scanner_count(self):
        """Return the fluorescence count at the current position."""
        
    def scanner_set_voltage(self, voltage):
        """Set the scanner voltage for analog control."""
        
    def close(self):
        """Close the scanning probe connection."""
        
    def configure_scanner(self, settings):
        """Configure the scanner parameters."""
        
    def get_scanner_axes(self):
        """Return the available scanner axes."""
        
    def set_up_scanner(self):
        """Set up the scanner hardware for operation."""
```

## Future Development

The NV-Center Quantum Computer Simulator is designed to be extensible for future development in several directions:

### Planned Enhancements

1. **Multi-Spin Systems**
   - Simulation of coupled NV centers
   - NV-nuclear spin interactions
   - Dipolar coupling networks

2. **Advanced Quantum Protocols**
   - Quantum error correction
   - Dynamical decoupling sequences
   - Quantum gate implementation

3. **Performance Optimizations**
   - GPU acceleration via QuTiP-GPU
   - Parallel computing for multiple NV centers
   - Optimized algorithms for large Hilbert spaces

4. **Extended Physical Models**
   - Temperature-dependent effects
   - Charge state dynamics (NV⁻/NV⁰)
   - Strain and electric field effects
   - Hyperfine interactions with detailed nuclear spin bath

5. **Advanced GUI Integration**
   - Interactive experiment designer
   - Real-time simulation visualization
   - Parameter optimization tools

6. **Machine Learning Integration**
   - Automated parameter identification
   - Quantum control optimization
   - Experiment design automation

7. **Additional Hardware Interfaces**
   - Pulse programmer integration
   - FPGA controller simulation
   - Arbitrary waveform generator support

### Contribution Guidelines

Contributions to the NV-Center Quantum Computer Simulator are welcome. Please follow these guidelines:

1. Follow the coding style from the existing codebase
2. Add comprehensive unit tests for new features
3. Update documentation to reflect changes
4. Ensure compatibility with the Qudi framework
5. Optimize for both accuracy and performance

For major changes, please open an issue first to discuss proposed modifications.