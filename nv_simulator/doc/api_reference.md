# NV-Zentrum Simulator - API-Referenz

## NVCenterSimulator Klasse

Die Hauptklasse für die Quantendynamik-Simulation des NV-Zentrums.

### Konstruktor

```python
NVCenterSimulator()
```

Initialisiert den Simulator mit Standard-Parametern.

### Eigenschaften

| Eigenschaft     | Typ           | Beschreibung                                     |
|-----------------|---------------|--------------------------------------------------|
| `D`             | float         | Nullfeldaufspaltung in Hz (Standard: 2.87e9)     |
| `E`             | float         | Strain-Aufspaltung in Hz (Standard: 0)           |
| `B_field`       | numpy.ndarray | Magnetfeldvektor [Bx, By, Bz] in Tesla          |
| `mw_freq`       | float         | Mikrowellenfrequenz in Hz                        |
| `mw_ampl`       | float         | Mikrowellenamplitude (Rabi-Frequenz) in Hz       |
| `mw_phase`      | float         | Mikrowellenphase in Rad                          |
| `laser_power`   | float         | Laserleistung (relative Einheit, 0-1)            |
| `t_max`         | float         | Maximale Simulationszeit in ns                   |
| `steps`         | int           | Anzahl der Zeitschritte für die Simulation       |
| `times`         | numpy.ndarray | Zeitachse der Simulation in Sekunden             |
| `results`       | qutip.Result  | Ergebnisse der Quantendynamik-Simulation         |
| `populations`   | dict          | Populationen der Zustände als Dictionary         |
| `initial_state` | qutip.Qobj    | Anfangszustand für die Simulation                |

### Methoden

#### Konfigurationsmethoden

```python
set_magnetic_field(Bx, By, Bz)
```
Setzt das externe Magnetfeld.
- `Bx, By, Bz`: Magnetfeldkomponenten in Tesla

```python
set_microwave(frequency, amplitude, phase=0.0)
```
Konfiguriert die Mikrowellenparameter.
- `frequency`: Mikrowellenfrequenz in Hz
- `amplitude`: Mikrowellenamplitude (Rabi-Frequenz) in Hz
- `phase`: Mikrowellenphase in Rad (optional)

```python
set_laser(power)
```
Konfiguriert die Laserleistung.
- `power`: Laserleistung (relative Einheit zwischen 0 und 1)

```python
set_time_parameters(t_max, steps)
```
Setzt die Zeitparameter für die Simulation.
- `t_max`: Maximale Simulationszeit in ns
- `steps`: Anzahl der Zeitschritte

```python
set_initial_state(state='ms0')
```
Setzt den Anfangszustand für die Simulation.
- `state`: Anfangszustand ('ms-1', 'ms0', 'ms+1' oder 'superposition')

#### Simulationsmethoden

```python
run_simulation()
```
Führt die Quantendynamik-Simulation des NV-Zentrums durch.

```python
simulate_odmr(freq_range, num_points=101, averaging=1)
```
Simuliert ein ODMR-Spektrum (Optically Detected Magnetic Resonance).
- `freq_range`: Tuple (min_freq, max_freq) für den Frequenzbereich in Hz
- `num_points`: Anzahl der Frequenzpunkte
- `averaging`: Anzahl der Wiederholungen für Mittelung
- **Rückgabe**: Dict mit 'frequencies' und 'pl_signal'

```python
simulate_rabi(mw_amps, pulse_duration)
```
Simuliert Rabi-Oszillationen bei verschiedenen Mikrowellenamplituden.
- `mw_amps`: Liste von Mikrowellenamplituden in Hz
- `pulse_duration`: Pulsdauer in ns
- **Rückgabe**: Dict mit 'amplitudes' und 'ms0_population'

#### Visualisierungsmethoden

```python
plot_results(save_path=None)
```
Visualisiert die Simulationsergebnisse (Zustandspopulationen über Zeit).
- `save_path`: Optional; Pfad zum Speichern der Grafik

```python
plot_odmr(odmr_data, save_path=None)
```
Visualisiert ein ODMR-Spektrum.
- `odmr_data`: Dict mit 'frequencies' und 'pl_signal'
- `save_path`: Optional; Pfad zum Speichern der Grafik

```python
plot_rabi(rabi_data, save_path=None)
```
Visualisiert Rabi-Oszillationen.
- `rabi_data`: Dict mit 'amplitudes' und 'ms0_population'
- `save_path`: Optional; Pfad zum Speichern der Grafik

#### Dateioperationen

```python
save_results(filename)
```
Speichert die Simulationsergebnisse als JSON-Datei.
- `filename`: Pfad zur Ausgabedatei

### Interne Methoden

```python
_create_hamiltonian()
```
Erzeugt den Hamiltonian für das NV-Zentrum.
- **Rückgabe**: qutip.Qobj: Hamiltonian-Operator

```python
_simulate_dynamics()
```
Führt die Zeitentwicklung des Quantensystems durch.
- **Rückgabe**: qutip.Result: Ergebnis der Zeitentwicklung

## Hardware-Modulklassen

### MicrowaveSimulator Klasse

Implementiert die Qudi MicrowaveInterface-Schnittstelle für simulierte Mikrowellenoperationen.

```python
MicrowaveSimulator()
```

#### Wichtige Methoden

```python
set_cw(frequency, power)
```
Konfiguriert den CW-Mikrowellenausgang.
- `frequency`: Frequenz in Hz
- `power`: Leistung in dBm

```python
cw_on()
```
Schaltet den CW-Mikrowellenausgang ein.

```python
configure_scan(power, frequencies, mode, sample_rate)
```
Konfiguriert einen Frequenzscan.
- `power`: Leistung in dBm
- `frequencies`: Frequenzen (Jump-Liste oder Tuple für gleichmäßigen Sweep)
- `mode`: SamplingOutputMode
- `sample_rate`: Abtastrate in Hz

```python
start_scan()
```
Startet den konfigurierten Mikrowellenscan.

```python
reset_scan()
```
Setzt den laufenden Scan zurück.

```python
off()
```
Schaltet jeden Mikrowellenausgang aus.

### LaserSimulator Klasse

Implementiert die Qudi SimpleLaserInterface-Schnittstelle für simulierte Laseroperationen.

```python
LaserSimulator()
```

#### Wichtige Methoden

```python
get_power()
```
Gibt die aktuelle Laserleistung in Watt zurück.

```python
set_power(power)
```
Setzt den Leistungssollwert in Watt.

```python
get_current()
```
Gibt den aktuellen Laserstrom zurück.

```python
set_current(current)
```
Setzt den Stromsollwert.

```python
get_laser_state()
```
Gibt den aktuellen Laserzustand zurück.

```python
set_laser_state(state)
```
Setzt den Laserzustand (EIN/AUS).

```python
get_shutter_state()
```
Gibt den aktuellen Shutter-Zustand zurück.

```python
set_shutter_state(state)
```
Setzt den Shutter-Zustand (OFFEN/GESCHLOSSEN).

### ScanningProbeSimulator Klasse

Implementiert die Qudi ScanningProbeInterface-Schnittstelle für simulierte konfokale Mikroskopie.

```python
ScanningProbeSimulator()
```

#### Wichtige Methoden

```python
scanner_axes
```
Dictionary der Scanner-Achsen-Metadaten.

```python
scanner_channels
```
Dictionary der Scanner-Kanal-Metadaten.

```python
scanner_constraints
```
Scanner-Einschränkungen für diese Hardware.

```python
scanner_position
```
Aktuelle Scanner-Position in Achseneinheiten.

```python
move_scanner(position)
```
Bewegt den Scanner zu einer bestimmten Position in Achseneinheiten.

```python
configure_scan(settings)
```
Konfiguriert den Scanner für einen Scan mit den angegebenen Einstellungen.

```python
start_scan(settings, back_settings=None)
```
Startet einen neuen Scan mit den angegebenen Einstellungen.

```python
stop_scan()
```
Stoppt den aktuell laufenden Scan.

```python
current_scan_data()
```
Gibt die aktuellen Scandaten zurück.

### NV-Zentrum-Operationen

```python
get_nv_center_at_position(position)
```
Findet das nächste NV-Zentrum an einer gegebenen Position.
- `position`: Dictionary mit Positionskoordinaten
- **Rückgabe**: NV-Zentrum-Objekt oder None