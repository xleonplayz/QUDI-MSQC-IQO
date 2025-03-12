# NV-Zentrum Simulator - Benutzerhandbuch

## Übersicht

Der NV-Zentrum Quantencomputer-Simulator ermöglicht die Simulation der Quantendynamik eines NV-Zentrums (Stickstoff-Fehlstellen-Zentrum) im Diamanten unter verschiedenen experimentellen Bedingungen. Der Simulator kann in zwei Modi verwendet werden:

1. **Als eigenständiges Kommandozeilenprogramm** für schnelle Simulationen und Tests
2. **Als integrierter Teil des Qudi-Frameworks** für komplexe Experimente mit simulierter Hardware

Dieses Handbuch erklärt die Verwendung beider Modi.

## Eigenständige Verwendung

### Installation

1. Installieren Sie die erforderlichen Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

2. Stellen Sie sicher, dass das Hauptskript ausführbar ist:
   ```bash
   chmod +x nv_simulator.py
   ```

### Grundlegende Kommandos

Der Simulator unterstützt drei Simulationstypen:

#### 1. Quantendynamik-Simulation

```bash
./nv_simulator.py --type dynamics \
                  --mw-freq 2.87e9 \
                  --mw-ampl 10e6 \
                  --t-max 1000 \
                  --initial ms0
```

Optionen:
- `--mw-freq`: Mikrowellenfrequenz in Hz
- `--mw-ampl`: Mikrowellenamplitude (Rabi-Frequenz) in Hz
- `--laser`: Laserleistung (relative Einheit, 0-1)
- `--t-max`: Maximale Simulationszeit in ns
- `--steps`: Anzahl der Zeitschritte
- `--initial`: Anfangszustand (`ms-1`, `ms0`, `ms+1`, `superposition`)

#### 2. ODMR-Spektrum-Simulation

```bash
./nv_simulator.py --type odmr \
                  --freq-min 2.77e9 \
                  --freq-max 2.97e9 \
                  --freq-points 101
```

Optionen:
- `--freq-min`: Minimale Frequenz für ODMR in Hz
- `--freq-max`: Maximale Frequenz für ODMR in Hz
- `--freq-points`: Anzahl der Frequenzpunkte

#### 3. Rabi-Oszillationen-Simulation

```bash
./nv_simulator.py --type rabi \
                  --mw-amp-min 0 \
                  --mw-amp-max 30e6 \
                  --mw-amp-points 31 \
                  --pulse-duration 500
```

Optionen:
- `--mw-amp-min`: Minimale Mikrowellenamplitude in Hz
- `--mw-amp-max`: Maximale Mikrowellenamplitude in Hz
- `--mw-amp-points`: Anzahl der Amplitudenpunkte
- `--pulse-duration`: Pulsdauer in ns

### Gemeinsame Optionen für alle Simulationstypen

- `--D`: Nullfeldaufspaltung in Hz (Standard: 2.87e9)
- `--E`: Strain-Aufspaltung in Hz (Standard: 0)
- `--Bx`, `--By`, `--Bz`: Magnetfeldkomponenten in Tesla
- `--save-plot`: Pfad zum Speichern der generierten Grafik
- `--save-data`: Pfad zum Speichern der Simulationsdaten im JSON-Format

### Beispiele

1. Simulation mit externem Magnetfeld:
   ```bash
   ./nv_simulator.py --type dynamics --Bz 0.02 --mw-freq 2.9e9 --mw-ampl 5e6 --t-max 1000
   ```

2. ODMR-Spektrum mit Strain-Effekt:
   ```bash
   ./nv_simulator.py --type odmr --E 5e6 --freq-min 2.77e9 --freq-max 2.97e9
   ```

3. Speichern von ODMR-Daten und Grafik:
   ```bash
   ./nv_simulator.py --type odmr --freq-min 2.77e9 --freq-max 2.97e9 --save-data odmr_data.json --save-plot odmr_plot.png
   ```

## Integration mit Qudi

### Konfiguration

Um den Simulator mit Qudi zu verwenden, müssen Sie die folgenden Einträge zu Ihrer Qudi-Konfigurationsdatei hinzufügen:

```yaml
hardware:
    mw_source_simulator:
        module.Class: 'nv_simulator.hardware.microwave_simulator.MicrowaveSimulator'
    
    laser_simulator:
        module.Class: 'nv_simulator.hardware.laser_simulator.LaserSimulator'
    
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

logic:
    laser_logic:
        module.Class: 'laser_logic.LaserLogic'
        connect:
            laser: laser_simulator
    
    scanning_probe_logic:
        module.Class: 'scanning_probe_logic.ScanningProbeLogic'
        connect:
            scanner: scanning_probe_simulator
    
    odmr_logic:
        module.Class: 'odmr_logic.ODMRLogic'
        connect:
            microwave1: mw_source_simulator
            fitlogic: fit_logic
```

Eine vollständige Beispielkonfiguration finden Sie in der Datei `example_config.yml`.

### Verwendung der Simulationsmodule

Nach der Konfiguration können Sie die Simulationshardware genauso wie echte Hardware in Qudi verwenden. Die Simulatoren reagieren auf dieselben Befehle und geben realistische simulierte Daten zurück.

#### Mikrowellenmodul (MicrowaveSimulator)

- Unterstützt CW-Modus und Frequenzscan-Modus
- Reagiert auf Standardbefehle wie `set_cw()`, `cw_on()`, `configure_scan()`, etc.
- Die simulierten ODMR-Spektren berücksichtigen Nullfeldaufspaltung, Zeeman-Effekt und Strain-Aufspaltung

#### Lasermodul (LaserSimulator)

- Unterstützt Leistungs- und Stromsteuerung
- Implementiert Ein/Aus-Steuerung und Shutter-Kontrolle
- Simuliert realistische Temperaturschwankungen

#### Scanning-Probe-Modul (ScanningProbeSimulator)

- Erzeugt realistische konfokale Bilder mit simulierten NV-Zentren
- Unterstützt 2D- und 3D-Scans
- Simuliert PSF (Point Spread Function) für realistisches Fluoreszenzprofil

## Programmierung und Erweiterung

### Verwendung des NVCenterSimulator in Ihrem eigenen Code

Sie können den Simulator-Kern direkt in Ihren eigenen Python-Code einbinden:

```python
from nv_simulator import NVCenterSimulator

# Simulator initialisieren
sim = NVCenterSimulator()

# Parameter konfigurieren
sim.D = 2.87e9  # Nullfeldaufspaltung
sim.set_magnetic_field(0, 0, 0.01)  # 10 mT entlang z-Achse
sim.set_microwave(2.87e9, 10e6)  # Mikrowellenfrequenz und -amplitude
sim.set_laser(0.5)  # Laserleistung (50%)

# Simulation ausführen
sim.run_simulation()

# Ergebnisse visualisieren
sim.plot_results()
```

### Anpassung der Simulation

Wenn Sie das Modell erweitern möchten, können Sie folgende Methoden modifizieren:

- `_create_hamiltonian()`: Ändern Sie den Hamiltonian, um zusätzliche Wechselwirkungen zu berücksichtigen
- `_simulate_dynamics()`: Passen Sie die Quantendynamik-Simulation an (z.B. andere Relaxationszeiten)
- `simulate_odmr()`: Modifizieren Sie die ODMR-Simulationsparameter
- `simulate_rabi()`: Passen Sie die Rabi-Oszillationen-Simulation an

## Fehlerbehebung

### Häufige Fehler

1. **ImportError: No module named 'qutip'**
   - Stellen Sie sicher, dass QuTiP installiert ist: `pip install qutip`

2. **Simulationsergebnisse entsprechen nicht den Erwartungen**
   - Überprüfen Sie die verwendeten Einheiten (alle Frequenzen in Hz, Zeit in s oder ns)
   - Erhöhen Sie die Zeitauflösung durch Erhöhung von `steps`

3. **Bei Qudi-Integration werden Simulationsmodule nicht gefunden**
   - Stellen Sie sicher, dass der Simulator-Pfad im Python-Suchpfad liegt
   - Überprüfen Sie die Klassen-Pfade in der Qudi-Konfigurationsdatei

### Testen

Der Simulator enthält eine umfangreiche Testsuite, die Sie ausführen können, um sicherzustellen, dass alles korrekt funktioniert:

```bash
cd test
./run_tests.py
```

Für spezifische Tests:
```bash
./run_tests.py --core-only     # Nur Kernsimulator testen
./run_tests.py --integration-only   # Nur Qudi-Integration testen
./run_tests.py --coverage      # Testabdeckung messen
```

## Weitere Ressourcen

- [README.md](../README.md): Allgemeine Übersicht über den Simulator
- [examples.sh](../examples.sh): Beispielskripte zur Veranschaulichung grundlegender Funktionen
- [example_config.yml](../example_config.yml): Beispiel-Qudi-Konfiguration