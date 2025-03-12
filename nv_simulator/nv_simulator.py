#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NV-Zentrum Quantencomputer-Simulator

Dieses Skript simuliert die Quantendynamik eines NV-Zentrums im Diamanten
unter Einfluss verschiedener Manipulationen (Mikrowellen, Laser).

Autor: Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis, tensor, sigmaz, sigmax, sigmay, identity, mesolve
import argparse
import json


class NVCenterSimulator:
    """Simulator für die Quantendynamik eines NV-Zentrums im Diamanten."""
    
    def __init__(self):
        """Initialisiert den NV-Zentrum-Simulator mit Standardparametern."""
        # Physikalische Konstanten
        self.h = 6.626e-34  # Planck-Konstante in J·s
        self.mu_B = 9.274e-24  # Bohr-Magneton in J/T
        self.g_e = 2.0023  # g-Faktor des Elektrons
        
        # NV-Zentrum Parameter
        self.D = 2.87e9  # Nullfeldaufspaltung in Hz
        self.E = 0  # Strain-Aufspaltung in Hz
        
        # Simulationsparameter
        self.B_field = np.array([0, 0, 0])  # Magnetfeld in [Bx, By, Bz] (Tesla)
        self.mw_freq = 2.87e9  # Mikrowellenfrequenz in Hz
        self.mw_ampl = 0.0  # Mikrowellenamplitude (Rabi-Frequenz) in Hz
        self.mw_phase = 0.0  # Mikrowellenphase in Rad
        self.laser_power = 0.0  # Laserleistung (relative Einheit)
        
        # Hilbert-Raum: |ms=-1⟩, |ms=0⟩, |ms=+1⟩
        self.dims = [[3], [1]]
        
        # Eigenzustände des NV-Zentrums
        self.ms_m1 = basis(3, 0)  # |ms=-1⟩
        self.ms_0 = basis(3, 1)   # |ms=0⟩
        self.ms_p1 = basis(3, 2)  # |ms=+1⟩
        
        # Anfangszustand (|ms=0⟩)
        self.initial_state = self.ms_0
        
        # Zeitparameter
        self.t_max = 1000  # Maximale Simulationszeit in ns
        self.steps = 1000  # Anzahl der Zeitschritte
        self.times = np.linspace(0, self.t_max*1e-9, self.steps)  # Zeitachse in s
        
        # Ergebnisse
        self.results = None
        self.populations = None
    
    def set_magnetic_field(self, Bx, By, Bz):
        """Setzt das externe Magnetfeld.
        
        Args:
            Bx, By, Bz: Magnetfeldkomponenten in Tesla
        """
        self.B_field = np.array([Bx, By, Bz])
    
    def set_microwave(self, frequency, amplitude, phase=0.0):
        """Konfiguriert die Mikrowellenparameter.
        
        Args:
            frequency: Mikrowellenfrequenz in Hz
            amplitude: Mikrowellenamplitude (Rabi-Frequenz) in Hz
            phase: Mikrowellenphase in Rad (optional)
        """
        self.mw_freq = frequency
        self.mw_ampl = amplitude
        self.mw_phase = phase
    
    def set_laser(self, power):
        """Konfiguriert die Laserleistung.
        
        Args:
            power: Laserleistung (relative Einheit zwischen 0 und 1)
        """
        self.laser_power = max(0, min(1, power))  # Beschränkung auf [0,1]
    
    def set_time_parameters(self, t_max, steps):
        """Setzt die Zeitparameter für die Simulation.
        
        Args:
            t_max: Maximale Simulationszeit in ns
            steps: Anzahl der Zeitschritte
        """
        self.t_max = t_max
        self.steps = steps
        self.times = np.linspace(0, self.t_max*1e-9, self.steps)  # Zeitachse in s
    
    def set_initial_state(self, state='ms0'):
        """Setzt den Anfangszustand für die Simulation.
        
        Args:
            state: Anfangszustand ('ms-1', 'ms0', 'ms+1' oder 'superposition')
        """
        if state == 'ms-1':
            self.initial_state = self.ms_m1
        elif state == 'ms0':
            self.initial_state = self.ms_0
        elif state == 'ms+1':
            self.initial_state = self.ms_p1
        elif state == 'superposition':
            # Überlagerungszustand (|ms=0⟩ + |ms=-1⟩)/√2
            self.initial_state = (self.ms_0 + self.ms_m1).unit()
        else:
            print(f"Unbekannter Zustand '{state}'. Verwende |ms=0⟩.")
            self.initial_state = self.ms_0
    
    def _create_hamiltonian(self):
        """Erzeugt den Hamiltonian für das NV-Zentrum.
        
        Returns:
            qutip.Qobj: Hamiltonian-Operator
        """
        # Spin-Operatoren für ein Spin-1-System
        # Für Operatoren ist dim [[n], [n]], nicht [[n], [1]]
        Sx = 1/np.sqrt(2) * Qobj([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dims=[[3], [3]])
        Sy = 1j/np.sqrt(2) * Qobj([[0, -1, 0], [1, 0, -1], [0, 1, 0]], dims=[[3], [3]])
        Sz = Qobj([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dims=[[3], [3]])
        
        # Nullfeldaufspaltung
        H_ZFS = self.D * (Sz*Sz - 2/3 * identity(3))
        
        # Strain-Term
        H_strain = self.E * (Sx*Sx - Sy*Sy)
        
        # Zeeman-Aufspaltung durch externes Magnetfeld
        H_zeeman = self.g_e * self.mu_B * (
            self.B_field[0] * Sx + 
            self.B_field[1] * Sy + 
            self.B_field[2] * Sz
        ) / self.h  # Umrechnung in Hz
        
        # Mikrowellen-Anteil (Rotating Wave Approximation)
        mw_drive_x = self.mw_ampl * np.cos(self.mw_phase) * Sx
        mw_drive_y = self.mw_ampl * np.sin(self.mw_phase) * Sy
        H_mw = mw_drive_x + mw_drive_y
        
        # Gesamter Hamiltonian (in der rotierenden Basis mit der Mikrowellenfrequenz)
        delta = self.mw_freq - self.D  # Verstimmung
        H_tot = H_ZFS + H_strain + H_zeeman - delta * Sz*Sz + H_mw
        
        return H_tot
    
    def _simulate_dynamics(self):
        """Führt die Zeitentwicklung des Quantensystems durch.
        
        Returns:
            qutip.Result: Ergebnis der Zeitentwicklung
        """
        # Hamiltonian erstellen
        H = self._create_hamiltonian()
        
        # Kollaps-Operatoren für Dekohärenz und Relaxation
        # Einfaches Modell für Spin-Relaxation (T1) und Dekohärenz (T2)
        T1 = 5e-3  # Spin-Relaxation in s
        T2 = 1e-6  # Spin-Dekohärenz in s
        
        c_ops = []
        
        # T1-Relaxation: |ms=±1⟩ -> |ms=0⟩
        if T1 > 0:
            rate = 1/T1
            c_ops.append(np.sqrt(rate) * self.ms_0 * self.ms_m1.dag())
            c_ops.append(np.sqrt(rate) * self.ms_0 * self.ms_p1.dag())
        
        # T2-Dekohärenz (Phasenverlust)
        if T2 > 0:
            rate = 1/T2
            # Korrigiere die Dimensions-Angabe für sz
            sz = Qobj([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dims=[[3], [3]])
            c_ops.append(np.sqrt(rate) * sz)
        
        # Laser-Effekt (vereinfacht): Optisches Pumpen von |ms=±1⟩ in |ms=0⟩
        if self.laser_power > 0:
            laser_rate = self.laser_power * 1e7  # Skalarrate für optisches Pumpen
            c_ops.append(np.sqrt(laser_rate) * self.ms_0 * self.ms_m1.dag())
            c_ops.append(np.sqrt(laser_rate) * self.ms_0 * self.ms_p1.dag())
        
        # Erwartungswerte berechnen: Populationen der Zustände
        e_ops = [
            self.ms_m1 * self.ms_m1.dag(),  # P(ms=-1)
            self.ms_0 * self.ms_0.dag(),    # P(ms=0)
            self.ms_p1 * self.ms_p1.dag()   # P(ms=+1)
        ]
        
        # Zeitentwicklung durchführen
        result = mesolve(H, self.initial_state, self.times, c_ops, e_ops)
        
        return result
    
    def run_simulation(self):
        """Führt die NV-Zentrum-Simulation aus."""
        print("Starte Simulation des NV-Zentrums...")
        
        # Simulation durchführen
        self.results = self._simulate_dynamics()
        
        # Populationen extrahieren
        self.populations = {
            'ms-1': self.results.expect[0],
            'ms0': self.results.expect[1],
            'ms+1': self.results.expect[2],
            'time': self.times * 1e9  # Umrechnung in ns
        }
        
        print("Simulation abgeschlossen!")
    
    def plot_results(self, save_path=None):
        """Visualisiert die Simulationsergebnisse.
        
        Args:
            save_path: Optional; Pfad zum Speichern der Grafik
        """
        if self.results is None:
            print("Keine Simulationsergebnisse verfügbar. Bitte führen Sie zuerst run_simulation() aus.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.populations['time'], self.populations['ms-1'], 'r-', label='|ms=-1⟩')
        plt.plot(self.populations['time'], self.populations['ms0'], 'g-', label='|ms=0⟩')
        plt.plot(self.populations['time'], self.populations['ms+1'], 'b-', label='|ms=+1⟩')
        
        plt.xlabel('Zeit (ns)')
        plt.ylabel('Populationen')
        plt.title('NV-Zentrum Quantendynamik')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Grafik gespeichert unter: {save_path}")
        else:
            plt.show()
    
    def save_results(self, filename):
        """Speichert die Simulationsergebnisse als JSON-Datei.
        
        Args:
            filename: Pfad zur Ausgabedatei
        """
        if self.results is None:
            print("Keine Simulationsergebnisse verfügbar. Bitte führen Sie zuerst run_simulation() aus.")
            return
        
        data = {
            'parameters': {
                'D': self.D,
                'E': self.E,
                'B_field': self.B_field.tolist(),
                'mw_freq': self.mw_freq,
                'mw_ampl': self.mw_ampl,
                'mw_phase': self.mw_phase,
                'laser_power': self.laser_power,
                't_max': self.t_max,
                'steps': self.steps
            },
            'results': {
                'time': self.populations['time'].tolist(),
                'ms-1': self.populations['ms-1'].tolist(),
                'ms0': self.populations['ms0'].tolist(),
                'ms+1': self.populations['ms+1'].tolist()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Ergebnisse gespeichert unter: {filename}")

    def simulate_odmr(self, freq_range, num_points=101, averaging=1):
        """Simuliert ein ODMR-Spektrum (Optically Detected Magnetic Resonance).
        
        Args:
            freq_range: Tuple (min_freq, max_freq) für den Frequenzbereich in Hz
            num_points: Anzahl der Frequenzpunkte
            averaging: Anzahl der Wiederholungen für Mittelung
            
        Returns:
            Dict mit Frequenzen und PL-Signal
        """
        print(f"Simuliere ODMR-Spektrum von {freq_range[0]/1e6:.2f} MHz bis {freq_range[1]/1e6:.2f} MHz...")
        
        frequencies = np.linspace(freq_range[0], freq_range[1], num_points)
        pl_signal = np.zeros(num_points)
        
        # Speichere ursprüngliche Parameter
        orig_mw_freq = self.mw_freq
        orig_mw_ampl = self.mw_ampl
        orig_t_max = self.t_max
        orig_steps = self.steps
        
        # Simulationsparameter für ODMR
        self.mw_ampl = 5e6  # 5 MHz Rabi-Frequenz
        self.t_max = 2000   # 2000 ns für jeden Punkt
        self.steps = 100
        self.times = np.linspace(0, self.t_max*1e-9, self.steps)
        self.set_initial_state('ms0')
        
        # Für jede Frequenz
        for i, freq in enumerate(frequencies):
            avg_pop = 0
            
            # Mehrere Durchläufe für Mittelung
            for _ in range(averaging):
                # Setze Mikrowellenfrequenz
                self.mw_freq = freq
                
                # Simulation ausführen
                result = self._simulate_dynamics()
                
                # Mittelung über die letzten 20% der Simulationszeit
                final_idx = int(0.8 * self.steps)
                avg_pop += np.mean(result.expect[1][final_idx:])  # ms=0 Population
            
            # Normieren
            pl_signal[i] = avg_pop / averaging
            
            # Fortschritt anzeigen
            if (i+1) % 10 == 0 or i == len(frequencies)-1:
                print(f"Fortschritt: {i+1}/{len(frequencies)} Frequenzen")
        
        # Parameter zurücksetzen
        self.mw_freq = orig_mw_freq
        self.mw_ampl = orig_mw_ampl
        self.t_max = orig_t_max
        self.steps = orig_steps
        self.times = np.linspace(0, self.t_max*1e-9, self.steps)
        
        print("ODMR-Simulation abgeschlossen!")
        
        return {
            'frequencies': frequencies,
            'pl_signal': pl_signal
        }
    
    def plot_odmr(self, odmr_data, save_path=None):
        """Visualisiert ein ODMR-Spektrum.
        
        Args:
            odmr_data: Dict mit 'frequencies' und 'pl_signal'
            save_path: Optional; Pfad zum Speichern der Grafik
        """
        frequencies = odmr_data['frequencies']
        pl_signal = odmr_data['pl_signal']
        
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies/1e6, pl_signal, 'b-')
        
        plt.xlabel('Frequenz (MHz)')
        plt.ylabel('PL-Intensität (a.u.)')
        plt.title('ODMR-Spektrum des NV-Zentrums')
        plt.grid(True)
        
        # Markiere die Resonanzen
        resonance_idx = np.argmin(pl_signal)
        resonance_freq = frequencies[resonance_idx]
        plt.axvline(x=resonance_freq/1e6, color='r', linestyle='--', 
                    label=f'Resonanz: {resonance_freq/1e6:.2f} MHz')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"ODMR-Grafik gespeichert unter: {save_path}")
        else:
            plt.show()
    
    def simulate_rabi(self, mw_amps, pulse_duration):
        """Simuliert Rabi-Oszillationen bei verschiedenen Mikrowellenamplituden.
        
        Args:
            mw_amps: Liste von Mikrowellenamplituden (Hz)
            pulse_duration: Pulsdauer in ns
            
        Returns:
            Dict mit Amplituden und Endzustandspopulationen
        """
        print(f"Simuliere Rabi-Oszillationen für {len(mw_amps)} verschiedene Amplituden...")
        
        # Speichere ursprüngliche Parameter
        orig_mw_ampl = self.mw_ampl
        orig_t_max = self.t_max
        orig_steps = self.steps
        
        # Setze Frequenz auf Resonanz
        self.mw_freq = self.D
        
        # Simulationsparameter für Rabi
        self.t_max = pulse_duration
        self.steps = 100
        self.times = np.linspace(0, self.t_max*1e-9, self.steps)
        self.set_initial_state('ms0')
        
        # Ergebnisse für alle Amplituden
        ms0_populations = []
        
        # Für jede Amplitude
        for i, amp in enumerate(mw_amps):
            # Setze Mikrowellenamplitude
            self.mw_ampl = amp
            
            # Simulation ausführen
            result = self._simulate_dynamics()
            
            # Finaler Zustand (ms=0 Population)
            final_pop = result.expect[1][-1]
            ms0_populations.append(final_pop)
            
            # Fortschritt anzeigen
            if (i+1) % 5 == 0 or i == len(mw_amps)-1:
                print(f"Fortschritt: {i+1}/{len(mw_amps)} Amplituden")
        
        # Parameter zurücksetzen
        self.mw_ampl = orig_mw_ampl
        self.t_max = orig_t_max
        self.steps = orig_steps
        self.times = np.linspace(0, self.t_max*1e-9, self.steps)
        
        print("Rabi-Simulation abgeschlossen!")
        
        return {
            'amplitudes': np.array(mw_amps),
            'ms0_population': np.array(ms0_populations)
        }
    
    def plot_rabi(self, rabi_data, save_path=None):
        """Visualisiert Rabi-Oszillationen.
        
        Args:
            rabi_data: Dict mit 'amplitudes' und 'ms0_population'
            save_path: Optional; Pfad zum Speichern der Grafik
        """
        amplitudes = rabi_data['amplitudes']
        ms0_pop = rabi_data['ms0_population']
        
        plt.figure(figsize=(10, 6))
        plt.plot(amplitudes/1e6, ms0_pop, 'g-o')
        
        plt.xlabel('Mikrowellenamplitude (MHz)')
        plt.ylabel('|ms=0⟩ Population')
        plt.title(f'Rabi-Oszillationen (Pulsdauer: {self.t_max} ns)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Rabi-Grafik gespeichert unter: {save_path}")
        else:
            plt.show()


def main():
    """Hauptfunktion für die Befehlszeileninteraktion."""
    parser = argparse.ArgumentParser(description='NV-Zentrum Quantencomputer-Simulator')
    
    # Simulationstyp
    parser.add_argument('--type', type=str, choices=['dynamics', 'odmr', 'rabi'], 
                        default='dynamics', help='Art der Simulation')
    
    # Parameter für alle Simulationstypen
    parser.add_argument('--D', type=float, default=2.87e9, help='Nullfeldaufspaltung in Hz')
    parser.add_argument('--E', type=float, default=0, help='Strain-Aufspaltung in Hz')
    parser.add_argument('--Bx', type=float, default=0, help='Magnetfeld in x-Richtung (Tesla)')
    parser.add_argument('--By', type=float, default=0, help='Magnetfeld in y-Richtung (Tesla)')
    parser.add_argument('--Bz', type=float, default=0, help='Magnetfeld in z-Richtung (Tesla)')
    
    # Parameter für Quantendynamik
    parser.add_argument('--mw-freq', type=float, help='Mikrowellenfrequenz in Hz')
    parser.add_argument('--mw-ampl', type=float, default=10e6, help='Mikrowellenamplitude in Hz')
    parser.add_argument('--laser', type=float, default=0, help='Laserleistung (0-1)')
    parser.add_argument('--t-max', type=float, default=1000, help='Maximale Simulationszeit in ns')
    parser.add_argument('--steps', type=int, default=1000, help='Anzahl der Zeitschritte')
    parser.add_argument('--initial', type=str, choices=['ms-1', 'ms0', 'ms+1', 'superposition'],
                       default='ms0', help='Anfangszustand')
    
    # Parameter für ODMR
    parser.add_argument('--freq-min', type=float, help='Minimale Frequenz für ODMR in Hz')
    parser.add_argument('--freq-max', type=float, help='Maximale Frequenz für ODMR in Hz')
    parser.add_argument('--freq-points', type=int, default=101, help='Anzahl der Frequenzpunkte für ODMR')
    
    # Parameter für Rabi
    parser.add_argument('--mw-amp-min', type=float, default=0, help='Minimale MW-Amplitude für Rabi in Hz')
    parser.add_argument('--mw-amp-max', type=float, default=50e6, help='Maximale MW-Amplitude für Rabi in Hz')
    parser.add_argument('--mw-amp-points', type=int, default=51, help='Anzahl der Amplitudenpunkte für Rabi')
    parser.add_argument('--pulse-duration', type=float, default=500, help='Pulsdauer für Rabi in ns')
    
    # Ausgabeoptionen
    parser.add_argument('--save-plot', type=str, help='Pfad zum Speichern der Grafik')
    parser.add_argument('--save-data', type=str, help='Pfad zum Speichern der Simulationsdaten')
    
    args = parser.parse_args()
    
    # Simulator initialisieren
    sim = NVCenterSimulator()
    
    # Allgemeine Parameter setzen
    sim.D = args.D
    sim.E = args.E
    sim.set_magnetic_field(args.Bx, args.By, args.Bz)
    
    if args.type == 'dynamics':
        # Dynamiksimulation konfigurieren
        if args.mw_freq is not None:
            sim.set_microwave(args.mw_freq, args.mw_ampl)
        else:
            sim.set_microwave(sim.D, args.mw_ampl)  # Resonanzfrequenz
        
        sim.set_laser(args.laser)
        sim.set_time_parameters(args.t_max, args.steps)
        sim.set_initial_state(args.initial)
        
        # Simulation ausführen
        sim.run_simulation()
        
        # Ergebnisse visualisieren/speichern
        if args.save_plot:
            sim.plot_results(args.save_plot)
        else:
            sim.plot_results()
        
        if args.save_data:
            sim.save_results(args.save_data)
    
    elif args.type == 'odmr':
        # ODMR-Simulation konfigurieren
        if args.freq_min is None or args.freq_max is None:
            # Standardbereich um D
            args.freq_min = sim.D - 200e6  # -200 MHz
            args.freq_max = sim.D + 200e6  # +200 MHz
        
        # ODMR-Simulation ausführen
        odmr_data = sim.simulate_odmr(
            (args.freq_min, args.freq_max),
            num_points=args.freq_points
        )
        
        # Ergebnisse visualisieren/speichern
        if args.save_plot:
            sim.plot_odmr(odmr_data, args.save_plot)
        else:
            sim.plot_odmr(odmr_data)
        
        if args.save_data:
            with open(args.save_data, 'w') as f:
                json.dump({
                    'frequencies': odmr_data['frequencies'].tolist(),
                    'pl_signal': odmr_data['pl_signal'].tolist()
                }, f, indent=4)
            print(f"ODMR-Daten gespeichert unter: {args.save_data}")
    
    elif args.type == 'rabi':
        # Rabi-Simulation konfigurieren
        mw_amps = np.linspace(args.mw_amp_min, args.mw_amp_max, args.mw_amp_points)
        
        # Rabi-Simulation ausführen
        rabi_data = sim.simulate_rabi(mw_amps, args.pulse_duration)
        
        # Ergebnisse visualisieren/speichern
        if args.save_plot:
            sim.plot_rabi(rabi_data, args.save_plot)
        else:
            sim.plot_rabi(rabi_data)
        
        if args.save_data:
            with open(args.save_data, 'w') as f:
                json.dump({
                    'amplitudes': rabi_data['amplitudes'].tolist(),
                    'ms0_population': rabi_data['ms0_population'].tolist()
                }, f, indent=4)
            print(f"Rabi-Daten gespeichert unter: {args.save_data}")


if __name__ == "__main__":
    main()