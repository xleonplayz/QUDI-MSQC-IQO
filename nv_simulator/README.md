# NV-Zentrum Quantencomputer-Simulator

## Überblick
Dieser Simulator ermöglicht die realistische Modellierung der Quantendynamik eines NV-Zentrums im Diamanten, einschließlich präziser Manipulationen durch Laser, Mikrowellen und Positionierungssysteme. Der Simulator kann sowohl eigenständig als auch in das Qudi-Framework integriert verwendet werden.

## Eigenständige Verwendung

### Installation

1. Installieren Sie die erforderlichen Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

2. Der Simulator ist dann über das Terminal verwendbar:
   ```bash
   cd nv_simulator
   ./nv_simulator.py --help
   ```

### Nutzung und Beispiele

Der Simulator unterstützt drei Hauptbetriebsarten:

1. **Quantendynamik-Simulation**
   ```bash
   ./nv_simulator.py --type dynamics --mw-freq 2.87e9 --mw-ampl 10e6 --t-max 1000
   ```

2. **ODMR-Spektren-Simulation**
   ```bash
   ./nv_simulator.py --type odmr --freq-min 2.77e9 --freq-max 2.97e9 --freq-points 101
   ```

3. **Rabi-Oszillationen-Simulation**
   ```bash
   ./nv_simulator.py --type rabi --mw-amp-min 0 --mw-amp-max 30e6 --pulse-duration 500
   ```

Für weitere Beispiele schauen Sie sich das Skript `examples.sh` an oder führen Sie es aus:
```bash
./examples.sh
```

## Integration mit Qudi

Der Simulator enthält Module, die in das Qudi-Framework integriert werden können, um die Laborumgebung komplett zu simulieren.

### Hardware-Module

1. **MicrowaveSimulator**: Simuliert eine Mikrowellenquelle für ODMR-Experimente
2. **LaserSimulator**: Simuliert einen 532nm Laser zur Anregung von NV-Zentren
3. **ScanningProbeSimulator**: Simuliert einen Konfokalmikroskop-Scanner für die Abbildung von NV-Zentren

### Konfiguration

Eine Beispiel-Konfigurationsdatei für Qudi befindet sich in `example_config.yml`. Integrieren Sie diese Einstellungen in Ihre Qudi-Konfiguration:

```yaml
hardware:
    mw_source_simulator:
        module.Class: 'nv_simulator.hardware.microwave_simulator.MicrowaveSimulator'
    laser_simulator:
        module.Class: 'nv_simulator.hardware.laser_simulator.LaserSimulator'
    scanning_probe_simulator:
        module.Class: 'nv_simulator.hardware.scanning_probe_simulator.ScanningProbeSimulator'
        options:
            position_ranges:
                x: [-50, 50]  # μm
                y: [-50, 50]  # μm
                z: [-25, 25]  # μm
            nv_density: 0.01  # NV centers per μm²
```

## Simulationsparameter

### Mikrowellen-Manipulation
- Frequenz: Typisch 2.87 GHz ± Zeeman-Verschiebung
- Amplitude: Bestimmt die Rabi-Frequenz
- Unterstützung von CW und Puls-Modus

### Laser-Manipulation
- Leistung: 0-250 mW
- Steuerung von Shutter und Laserleistung für optisches Pumpen

### Konfokale Abbildung
- 3D-Positionierung (x, y, z)
- Simulierte PSF für realistische Abbildung
- Zufällig verteilte NV-Zentren mit individuellen Eigenschaften

## Implementierungsdetails

Der Simulator basiert auf dem QuTiP-Framework (Quantum Toolbox in Python) für die Quantendynamik-Simulation und integriert diese mit der Qudi-Hardware-Abstraktion. Die Kombination ermöglicht realistische Simulationen von:

- ODMR-Spektren
- Rabi-Oszillationen
- Spin-Kohärenz
- Konfokalen Scans
- Laser-induzierter Fluoreszenz

## Erweiterungsmöglichkeiten

1. Integration von Quantum Espresso für atomistische Modellierung
2. Erweiterung um Mehrspin-Systeme für Quantengatter
3. Hinzufügen von Dekohärenz-Modellen für realistischere T1/T2-Simulation