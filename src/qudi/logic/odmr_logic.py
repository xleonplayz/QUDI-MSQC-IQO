# -*- coding: utf-8 -*-

"""
This file contains the Qudi Logic module base class.

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

import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from PySide2 import QtCore

from qudi.util.datafitting import FitContainer, FitConfigurationsModel
from qudi.core.module import LogicBase
from qudi.util.mutex import RecursiveMutex
from qudi.util.units import ScaledFloat
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.datastorage import TextDataStorage
from qudi.util.enums import SamplingOutputMode


class OdmrLogic(LogicBase):
    """
    This is the Logic class for CW ODMR measurements.

    example config for copy-paste:

    odmr_logic:
        module.Class: 'odmr_logic.OdmrLogic'
        connect:
            microwave: <microwave_name>
            data_scanner: <data_scanner_name>
        options:
            default_scan_mode: 'JUMP_LIST'  # optional
    """

    # declare connectors
    _microwave = Connector(name='microwave', interface='MicrowaveInterface')
    _data_scanner = Connector(name='data_scanner', interface='FiniteSamplingInputInterface')

    # declare config options
    _save_thumbnails = ConfigOption(name='save_thumbnails', default=True)
    _default_scan_mode = ConfigOption(name='default_scan_mode',
                                      default='JUMP_LIST',
                                      constructor=lambda x: SamplingOutputMode[x.upper()])

    # declare status variables
    _cw_frequency = StatusVar(name='cw_frequency', default=2870e6)
    _cw_power = StatusVar(name='cw_power', default=-np.inf)
    _scan_power = StatusVar(name='scan_power', default=-np.inf)
    _scan_frequency_ranges = StatusVar(name='scan_frequency_ranges',
                                       default=[(2820e6, 2920e6, 101)])
    _run_time = StatusVar(name='run_time', default=60)
    _scans_to_average = StatusVar(name='scans_to_average', default=0)
    _data_rate = StatusVar(name='data_rate', default=200)
    _oversampling_factor = StatusVar(name='oversampling_factor', default=1)
    _fit_configs = StatusVar(name='fit_configs', default=None)

    # Internal signals
    _sigNextLine = QtCore.Signal()

    # Update signals, e.g. for GUI module
    sigScanParametersUpdated = QtCore.Signal(dict)
    sigCwParametersUpdated = QtCore.Signal(dict)
    sigElapsedUpdated = QtCore.Signal(float, int)
    sigScanStateUpdated = QtCore.Signal(bool)
    sigCwStateUpdated = QtCore.Signal(bool)
    sigScanDataUpdated = QtCore.Signal()
    sigFitUpdated = QtCore.Signal(object, str, int)

    __default_fit_configs = (
        {'name'             : 'Gaussian Dip',
         'model'            : 'Gaussian',
         'estimator'        : 'Dip',
         'custom_parameters': None},

        {'name'             : 'Two Gaussian Dips',
         'model'            : 'DoubleGaussian',
         'estimator'        : 'Dips',
         'custom_parameters': None},

        {'name'             : 'Lorentzian Dip',
         'model'            : 'Lorentzian',
         'estimator'        : 'Dip',
         'custom_parameters': None},

        {'name'             : 'Two Lorentzian Dips',
         'model'            : 'DoubleLorentzian',
         'estimator'        : 'Dips',
         'custom_parameters': None},
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._threadlock = RecursiveMutex()
        self._elapsed_time = 0.0
        self._elapsed_sweeps = 0
        self.__estimated_lines = 0
        self._start_time = 0.0
        self._fit_container = None
        self._fit_config_model = None
        self._raw_data = None
        self._signal_data = None
        self._frequency_data = None
        self._fit_results = None
        self._is_second_scan = False  # Neue Variable, um den zweiten Scan zu kennzeichnen.


    def on_activate(self):
        """
        Initialisation performed during activation of the module.
        """
        # Recall status variables and check against constraints
        mw_constraints = self._microwave().constraints
        data_constraints = self._data_scanner().constraints

        self._cw_frequency = mw_constraints.frequency_in_range(self._cw_frequency)[1]
        self._cw_power = mw_constraints.power_in_range(self._cw_power)[1]
        self._scan_power = mw_constraints.power_in_range(self._scan_power)[1]
        self._run_time = max(1., self._run_time)
        self._scans_to_average = max(0, int(self._scans_to_average))
        self._oversampling_factor = max(1, int(self._oversampling_factor))
        for ii, freq_range in enumerate(self._scan_frequency_ranges):
            self._scan_frequency_ranges[ii] = (
                mw_constraints.frequency_in_range(freq_range[0])[1],
                mw_constraints.frequency_in_range(freq_range[1])[1],
                mw_constraints.scan_size_in_range(int(freq_range[2]))[1]
            )
        # ToDo: Check against data sampler constraints
        # self._data_rate =

        # Set up fit model and container
        self._fit_config_model = FitConfigurationsModel(parent=self)
        self._fit_config_model.load_configs(self._fit_configs)
        self._fit_container = FitContainer(parent=self, config_model=self._fit_config_model)

        # Elapsed measurement time and number of sweeps
        self._elapsed_time = 0.0
        self._elapsed_sweeps = 0
        self._start_time = 0.0
        self.__estimated_lines = 0

        # Initialize the ODMR data arrays (mean signal and sweep matrix)
        self._initialize_odmr_data()

        # Connect signals
        self._sigNextLine.connect(self._scan_odmr_line, QtCore.Qt.QueuedConnection)

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        # Stop measurement if it is still running
        self._sigNextLine.disconnect()
        if self.module_state() == 'locked':
            self.stop_odmr_scan()

    @_fit_configs.representer
    def __repr_fit_configs(self, value):
        configs = self.fit_config_model.dump_configs()
        if len(configs) < 1:
            configs = None
        return configs

    @_fit_configs.constructor
    def __constr_fit_configs(self, value):
        if not value:
            return self.__default_fit_configs
        return value

    def _initialize_odmr_data(self):
        """ Initializing the ODMR data arrays (signal and raw data matrix). """
        self._frequency_data = [np.linspace(*r) for r in self._scan_frequency_ranges]

        self._raw_data = dict()
        self._fit_results = dict()
        self._signal_data = dict()
        estimated_samples = self._run_time * self._data_rate
        samples_per_line = sum(freq_range[-1] for freq_range in self._scan_frequency_ranges)
        # Add 5% Safety; Minimum of 1 line
        self.__estimated_lines = max(1, int(1.05 * estimated_samples / samples_per_line))
        for channel in self._data_scanner().constraints.channel_names:
            self._raw_data[channel] = [
                np.full((freq_arr.size, self.__estimated_lines), np.nan) for freq_arr in
                self._frequency_data
            ]
            self._signal_data[channel] = [
                np.zeros(freq_arr.size) for freq_arr in self._frequency_data
            ]
            self._fit_results[channel] = [None] * len(self._frequency_data)

    def _calculate_signal_data(self):
        for channel, raw_data_list in self._raw_data.items():
            for range_index, raw_data in enumerate(raw_data_list):
                masked_raw_data = np.ma.masked_invalid(raw_data)
                if masked_raw_data.compressed().size == 0:
                    arr_size = self._frequency_data[range_index].size
                    self._signal_data[channel][range_index] = np.zeros(arr_size)
                elif self._scans_to_average > 0:
                    self._signal_data[channel][range_index] = np.mean(
                        masked_raw_data[:, :self._scans_to_average],
                        axis=1
                    ).compressed()
                    if self._signal_data[channel][range_index].size == 0:
                        arr_size = self._frequency_data[range_index].size
                        self._signal_data[channel][range_index] = np.zeros(arr_size)
                else:
                    self._signal_data[channel][range_index] = np.mean(masked_raw_data,
                                                                      axis=1).compressed()

    @property
    def fit_config_model(self):
        return self._fit_config_model

    @property
    def fit_container(self):
        return self._fit_container

    @property
    def fit_results(self):
        return self._fit_results.copy()

    @property
    def data_constraints(self):
        return self._data_scanner().constraints

    @property
    def microwave_constraints(self):
        return self._microwave().constraints

    @property
    def signal_data(self):
        return self._signal_data.copy()

    @property
    def raw_data(self):
        return self._raw_data.copy()

    @property
    def frequency_data(self):
        return self._frequency_data.copy()

    @property
    def scans_to_average(self):
        return self._scans_to_average

    @scans_to_average.setter
    def scans_to_average(self, number_of_scans):
        self.set_scans_to_average(number_of_scans)

    @QtCore.Slot(int)
    def set_scans_to_average(self, number_of_scans):
        """ Sets the number of scans to average for the sum of the data

        @param int number_of_scans: desired number of scans to average (0 means all)
        """
        with self._threadlock:
            scans_to_average = int(number_of_scans)
            if scans_to_average != self._scans_to_average:
                self._scans_to_average = scans_to_average
                self._calculate_signal_data()
                self.sigScanParametersUpdated.emit({'averaged_scans': self._scans_to_average})
                self.sigScanDataUpdated.emit()

    @property
    def runtime(self):
        return self._run_time

    @runtime.setter
    def runtime(self, new_runtime):
        self.set_runtime(new_runtime)

    @QtCore.Slot(object)
    def set_runtime(self, runtime):
        """ Sets the runtime for ODMR measurement

        @param float runtime: desired runtime in seconds
        """
        with self._threadlock:
            try:
                self._run_time = float(runtime)
            except (TypeError, ValueError):
                self.log.exception('set_runtime failed:')
            self.sigScanParametersUpdated.emit({'run_time': self._run_time})

    @property
    def scan_power(self):
        return self._scan_power

    @scan_power.setter
    def scan_power(self, value):
        self.set_scan_power(value)

    @QtCore.Slot(object)
    def set_scan_power(self, scan_power):
        """ Sets the runtime for ODMR measurement

        @param float scan_power: desired power for scans in dBm
        """
        with self._threadlock:
            try:
                self._scan_power = float(scan_power)
            except (TypeError, ValueError):
                self.log.exception('scan_power failed:')
            self.sigScanParametersUpdated.emit({'scan_power': self._scan_power})

    @property
    def frequency_ranges(self):
        return self._scan_frequency_ranges.copy()

    @QtCore.Slot(object, object, object, int)
    def set_frequency_range(self, start, stop, points, index):
        with self._threadlock:
            if self.module_state() != 'idle':
                self.log.error('Unable to set frequency range. ODMR scan in progress.')
            else:
                try:
                    new_range = (start, stop, points)
                    if new_range != self._scan_frequency_ranges[index]:
                        self._scan_frequency_ranges[index] = new_range
                        self._initialize_odmr_data()
                        self.sigScanDataUpdated.emit()
                except IndexError:
                    self.log.exception('Frequency range index is out of range.')
                except:
                    self.log.exception('Error while trying to set frequency range:')
            self.sigScanParametersUpdated.emit({'frequency_ranges': self.frequency_ranges})

    @property
    def frequency_range_count(self):
        return len(self._scan_frequency_ranges)

    @QtCore.Slot(int)
    def set_frequency_range_count(self, number_of_ranges):
        if number_of_ranges < 1:
            self.log.error('Number of frequency ranges can not be smaller than 1.')
            self.sigScanParametersUpdated.emit({'frequency_ranges': self.frequency_ranges})
            return

        with self._threadlock:
            if self.module_state() != 'idle':
                self.log.error('Unable to set frequency range count. ODMR scan in progress.')
                self.sigScanParametersUpdated.emit({'frequency_ranges': self.frequency_ranges})
                return

            number_diff = number_of_ranges - self.frequency_range_count
            if number_diff < 0:
                del self._scan_frequency_ranges[number_of_ranges:]
            elif number_diff > 0:
                constraints = self.microwave_constraints
                if constraints.mode_supported(SamplingOutputMode.JUMP_LIST):
                    new_range = self._scan_frequency_ranges[-1]
                    self._scan_frequency_ranges.extend([new_range] * number_diff)
                else:
                    self.log.error('Multiple frequency ranges not supported by ODMR scanner '
                                   '(no "JUMP_LIST" output mode).')
            if number_diff != 0:
                self._initialize_odmr_data()
                self.sigScanDataUpdated.emit()
                self.sigScanParametersUpdated.emit({'frequency_ranges': self.frequency_ranges})

    @property
    def data_rate(self):
        return self._data_rate

    @data_rate.setter
    def data_rate(self, rate):
        self.set_data_rate(rate)

    @QtCore.Slot(object)
    def set_data_rate(self, rate):
        """
        @param float rate: desired data rate in Hz
        """
        self.set_sample_rate(data_rate=rate)

    @property
    def oversampling(self):
        return self._oversampling_factor

    @oversampling.setter
    def oversampling(self, factor):
        self.set_oversampling(factor)

    @QtCore.Slot(int)
    def set_oversampling(self, factor):
        self.set_sample_rate(oversampling=factor)

    @QtCore.Slot(object, int)
    def set_sample_rate(self, data_rate=None, oversampling=None):
        """ Helper method to set data rate and oversampling factor simultaneously. This method
        should be used whenever possible in order to avoid out-of-range errors when setting these
        two settings sequentially.
        """
        if data_rate is None and oversampling is None:
            return
        with self._threadlock:
            # checks if scanner is still running
            if self.module_state() == 'locked':
                self.log.error('Unable to set sample rate. ODMR measurement in progress.')
            else:
                data_rate = self.data_rate if data_rate is None else float(data_rate)
                oversampling = self.oversampling if oversampling is None else max(1,
                                                                                  int(oversampling))
                if self.data_constraints.sample_rate_in_range(data_rate * oversampling)[0]:
                    self._data_rate = data_rate
                    self._oversampling_factor = oversampling
                else:
                    self.log.error('Unable to set sample rate. Resulting sample rate out of bounds '
                                   'for ODMR scanner constraints.')
            self.sigScanParametersUpdated.emit(
                {'data_rate': self._data_rate, 'oversampling': self._oversampling_factor}
            )

    @property
    def scan_parameters(self):
        params = {'data_rate': self._data_rate,
                  'oversampling': self._oversampling_factor,
                  'frequency_ranges': self.frequency_ranges,
                  'run_time': self._run_time,
                  'averaged_scans': self._scans_to_average,
                  'power': self._scan_power}
        return params

    @property
    def cw_parameters(self):
        return {'frequency': self._cw_frequency, 'power': self._cw_power}

    @QtCore.Slot(object, object)
    def set_cw_parameters(self, frequency, power):
        """ Set the desired new cw mode parameters.

        @param float frequency: frequency to set in Hz
        @param float power: power to set in dBm
        """
        with self._threadlock:
            try:
                constraints = self.microwave_constraints
                self._cw_frequency = constraints.frequency_in_range(frequency)[1]
                self._cw_power = constraints.power_in_range(power)[1]
            except:
                self.log.exception('Error while trying to set CW parameters:')
            self.sigCwParametersUpdated.emit(self.cw_parameters)

    @QtCore.Slot(bool)
    def toggle_cw_output(self, enable):
        with self._threadlock:
            microwave = self._microwave()
            # Return early if CW output is already in desired state
            if enable == (microwave.module_state() != 'idle' and not microwave.is_scanning):
                self.sigCwStateUpdated.emit(enable)
                return
            # Throw error and return early if CW output can not be turned on
            if enable and self.module_state() != 'idle':
                self.log.error('Unable to turn on microwave CW output. ODMR scan in progress.')
                return
            # Toggle microwave output
            try:
                if enable:
                    microwave.set_cw(power=self._cw_power, frequency=self._cw_frequency)
                    microwave.cw_on()
                else:
                    microwave.off()
            except:
                self.log.exception('Error while trying to toggle microwave CW output:')
            finally:
                self.sigCwStateUpdated.emit(
                    microwave.module_state() != 'idle' and not microwave.is_scanning
                )

    # Neue Signaldeklaration (auf Klassenebene, zusammen mit den anderen Signalen)
    sigScanFinished = QtCore.Signal()

    @QtCore.Slot()
    def stop_odmr_scan(self):
        """ Stoppt den ODMR-Scan und signalisiert, dass der Scan beendet wurde. """
        with self._threadlock:
            if self.module_state() == 'locked':
                self._microwave().off()
                self.module_state.unlock()
            self.sigScanStateUpdated.emit(False)
            # Signal, dass der Scan fertig ist
            self.sigScanFinished.emit()

    @QtCore.Slot(bool, bool)
    def toggle_odmr_scan(self, start, resume):
        """
        Startet (oder stoppt) den ODMR-Scan.
        Wenn start=True, wird der Scan gestartet und nach Beendigung automatisch ein zweiter Scan initiiert.
        """
        if start:
            # Oberfläche sperren
            self.sigScanStateUpdated.emit(True)
            # Scan-Wiederholungs-Zähler zurücksetzen
            self._scan_counter = 0
            # Verbinde das finished-Signal mit dem Slot, der den nächsten Scan startet
            self.sigScanFinished.connect(self._on_scan_finished)
            # Ersten Scan starten
            self.start_odmr_scan()
        else:
            self.stop_odmr_scan()
            self.sigScanStateUpdated.emit(False)
            try:
                self.sigScanFinished.disconnect(self._on_scan_finished)
            except Exception:
                pass

    def _on_scan_finished(self):
        self._scan_counter += 1
        if self._scan_counter == 1:
            # Nach dem ersten Scan: Frequenzbereich verfeinern
            for i in range(len(self._scan_frequency_ranges)):
                new_freq = self.refine_frequency_range_v4(i)
                self._frequency_data[i] = new_freq
            self._is_second_scan = True  # Kennzeichne, dass der 2. Scan gestartet wird
            self.start_odmr_scan()
        else:
            self.sigScanFinished.disconnect(self._on_scan_finished)
            self._scan_counter = 0
            self._is_second_scan = False




    def refine_frequency_range_v2(self, range_index, drop_threshold_factor=0.5, concentration_power=3):
        """
        Verfeinert den Frequenzbereich für den zweiten Scan. Dabei werden:
        - 50% der Punkte gleichmäßig über den gesamten Frequenzbereich verteilt,
        - 50% der Punkte in dem ermittelten Drop-Bereich konzentriert, sodass
            in den tiefsten (anomal tiefen) Punkten mehr Messpunkte liegen.
        
        Parameter:
        range_index: Index des Frequenzbereichs (entspricht einem Element in self._scan_frequency_ranges).
        drop_threshold_factor: Faktor, um den Drop-Bereich relativ zur Differenz zwischen Basislinie 
                                und Minimalwert zu definieren. Beispielsweise bedeutet 0.5,
                                dass als Drop-Bereich alle Punkte gelten, deren Signal
                                unter (Basislinie - 0.5*(Basislinie - Minimum)) liegt.
        concentration_power: Exponent, der bestimmt, wie stark die Punkte im Drop-Bereich konzentriert werden.
                            Höhere Werte bewirken, dass sich die Punkte stärker um den tiefsten Punkt sammeln.
        
        Rückgabe:
        new_freq: Der neu berechnete Frequenzarray für diesen Bereich.
        """
        # Ursprünglicher Frequenzbereich und zugehöriges Signal (hier exemplarisch für den ersten Kanal)
        freq = self._frequency_data[range_index]
        channel = list(self._signal_data.keys())[0]
        signal = self._signal_data[channel][range_index]
        
        total_points = len(freq)
        n_uniform = total_points // 2  # 50% gleichmäßig verteilt
        n_drop = total_points - n_uniform  # restliche 50% für den Drop-Bereich

        # Bestimme Basislinie (z. B. Median) und den minimalen Signalwert
        baseline = np.median(signal)
        min_val = np.min(signal)
        min_index = np.argmin(signal)
        
        # Definiere einen Schwellenwert, um den Drop-Bereich zu identifizieren
        threshold = baseline - drop_threshold_factor * (baseline - min_val)
        
        # Finde Indizes, bei denen das Signal unter dem Schwellenwert liegt
        drop_indices = np.where(signal < threshold)[0]
        if len(drop_indices) == 0:
            # Kein signifikanter Drop erkannt – gib den ursprünglichen Frequenzbereich zurück
            return freq

        # Bestimme den Drop-Bereich: von der ersten bis zur letzten Stelle, an der das Signal unter dem Schwellenwert liegt
        drop_start_idx = drop_indices[0]
        drop_end_idx = drop_indices[-1]
        drop_start_freq = freq[drop_start_idx]
        drop_end_freq = freq[drop_end_idx]
        
        # Aufteilen des Drop-Bereichs in zwei Segmente: links (bis zum Minimalpunkt) und rechts (ab Minimalpunkt)
        n_drop_left = n_drop // 2
        n_drop_right = n_drop - n_drop_left
        
        # Linker Teil: von drop_start_freq bis zum minimalen Punkt
        min_freq = freq[min_index]
        x_left = np.linspace(0, 1, n_drop_left)
        # Durch Anwenden einer Potenzfunktion werden die Punkte ungleich verteilt – höhere Konzentration nahe dem Minimum
        left_transformed = x_left**concentration_power
        drop_left = drop_start_freq + (min_freq - drop_start_freq) * left_transformed

        # Rechter Teil: von minimalem Punkt bis drop_end_freq
        x_right = np.linspace(0, 1, n_drop_right)
        right_transformed = x_right**concentration_power
        drop_right = min_freq + (drop_end_freq - min_freq) * right_transformed

        # Kombiniere beide Teile zu einem Drop-Raster
        drop_grid = np.concatenate((drop_left, drop_right))
        
        # Erzeuge ein gleichmäßig verteiltes Raster über den gesamten Frequenzbereich
        uniform_grid = np.linspace(freq[0], freq[-1], n_uniform)
        
        # Kombiniere die Raster und sortiere sie
        new_freq = np.sort(np.concatenate((uniform_grid, drop_grid)))
        
        # Optional: Falls die Gesamtzahl der Punkte nicht exakt total_points beträgt,
        # kann hier noch eine Anpassung erfolgen (z. B. durch Interpolation oder Resampling).
        if len(new_freq) > total_points:
            indices = np.linspace(0, len(new_freq) - 1, total_points).astype(int)
            new_freq = new_freq[indices]
        elif len(new_freq) < total_points:
            new_freq = np.linspace(new_freq[0], new_freq[-1], total_points)
        
        return new_freq



    def refine_frequency_range_v3(self, range_index, drop_fraction=0.15, drop_width_fraction=0.01):
        """
        Verfeinert den Frequenzbereich, indem:
        - 85% der Punkte gleichmäßig über den gesamten Bereich verteilt werden
        - 15% der Punkte in einem engen Intervall um den Tiefpunkt (minimaler Signalwert) konzentriert werden

        Parameter:
        range_index: Index des Frequenzbereichs (entspricht einem Element in self._scan_frequency_ranges)
        drop_fraction: Anteil der Gesamtpunkte, die im Drop-Bereich konzentriert werden (Standard 15%)
        drop_width_fraction: Breite des Drop-Bereichs als Anteil des gesamten Frequenzbereichs (Standard 1%)

        Rückgabe:
        new_freq: Der verfeinerte Frequenzarray
        """
        # Ursprünglicher Frequenzbereich
        freq = self._frequency_data[range_index]
        # Für das Signal nehmen wir exemplarisch den ersten Kanal
        channel = list(self._signal_data.keys())[0]
        signal = self._signal_data[channel][range_index]
        
        total_points = len(freq)
        N_drop = int(np.round(drop_fraction * total_points))
        N_uniform = total_points - N_drop

        # Gleichmäßig verteiltes Raster über den gesamten Frequenzbereich (85% der Punkte)
        uniform_grid = np.linspace(freq[0], freq[-1], N_uniform)

        # Bestimme den Tiefpunkt (minimaler Signalwert) im Messsignal
        min_index = np.argmin(signal)
        min_freq = freq[min_index]

        # Berechne die Breite des Drop-Bereichs als ein Anteil des Gesamtbereichs
        total_range = freq[-1] - freq[0]
        drop_width = drop_width_fraction * total_range
        # Definiere den Dropbereich (achte darauf, dass wir innerhalb des Gesamtbereichs bleiben)
        drop_start = max(freq[0], min_freq - drop_width / 2)
        drop_end = min(freq[-1], min_freq + drop_width / 2)

        # Erzeuge ein dichtes Raster im Dropbereich (15% der Punkte)
        drop_grid = np.linspace(drop_start, drop_end, N_drop)

        # Kombiniere beide Raster und sortiere sie
        new_freq = np.sort(np.concatenate((uniform_grid, drop_grid)))

        # Sollte es aufgrund von Rundungen oder Überschneidungen nicht genau total_points geben,
        # erzwingen wir hier ein finales Raster mit exakt total_points Elementen.
        if len(new_freq) != total_points:
            new_freq = np.linspace(new_freq[0], new_freq[-1], total_points)
        
        return new_freq

    def refine_frequency_range_v4(self, range_index, anomaly_threshold_factor=0.5):
        """
        Verfeinert den Frequenzbereich, indem:
        - 50% der Punkte gleichmäßig außerhalb des Anomaliebereichs verteilt werden und
        - 50% der Punkte gleichmäßig innerhalb des Anomaliebereichs gesetzt werden.
        
        Der Anomaliebereich wird anhand des Signals identifiziert:
        - Es wird eine Basislinie (z. B. der Median) ermittelt und der minimale Signalwert bestimmt.
        - Ein Schwellenwert wird definiert als:
                threshold = baseline - anomaly_threshold_factor * (baseline - min_val)
        - Alle Frequenzpunkte, bei denen das Signal unterhalb dieses Schwellenwerts liegt, gelten als Teil des Anomaliebereichs.
        
        Falls keine Anomalie gefunden wird, wird das ursprüngliche Frequenzraster zurückgegeben.
        
        Parameter:
        range_index: Index des Frequenzbereichs in self._frequency_data.
        anomaly_threshold_factor: Faktor zur Bestimmung des Schwellenwerts (Standard: 0.5).
        
        Rückgabe:
        new_freq: Das neu berechnete Frequenzarray.
        """
        # Ursprünglicher Frequenzbereich
        freq = self._frequency_data[range_index]
        # Beispielhaft nehmen wir das Signal des ersten Kanals
        channel = list(self._signal_data.keys())[0]
        signal = self._signal_data[channel][range_index]
        
        N_total = len(freq)
        # 50% der Punkte für den Anomaliebereich und 50% für den Rest
        N_anomaly = int(np.round(0.5 * N_total))
        N_non_anomaly = N_total - N_anomaly

        # Basislinie (Median) und Minimalwert des Signals ermitteln
        baseline = np.median(signal)
        min_val = np.min(signal)
        
        # Schwellenwert definieren: Punkte unterhalb dieses Wertes gelten als Anomalie
        threshold = baseline - anomaly_threshold_factor * (baseline - min_val)
        
        # Bestimme die Indizes, an denen das Signal unterhalb des Schwellenwerts liegt
        anomaly_indices = np.where(signal < threshold)[0]
        if len(anomaly_indices) == 0:
            # Falls keine Anomalie gefunden wird, gebe das ursprüngliche Raster zurück
            return freq
        
        # Der Anomaliebereich wird von der ersten bis zur letzten Stelle, an der das Signal unter dem Schwellenwert liegt, definiert
        anomaly_start = freq[anomaly_indices[0]]
        anomaly_end = freq[anomaly_indices[-1]]
        
        # Erzeuge ein Raster mit N_anomaly Punkten, gleichmäßig im Anomaliebereich
        anomaly_grid = np.linspace(anomaly_start, anomaly_end, N_anomaly)
        
        # Erzeuge das Raster für den nicht-anomalen Bereich, der aus zwei Intervallen besteht:
        #   Intervall 1: [freq[0], anomaly_start]
        #   Intervall 2: [anomaly_end, freq[-1]]
        L1 = anomaly_start - freq[0]
        L2 = freq[-1] - anomaly_end
        L_total = L1 + L2
        if L_total <= 0:
            new_freq = anomaly_grid
        else:
            # Punkteverteilung proportional zur Länge der beiden Intervalle
            N1 = int(np.round(N_non_anomaly * (L1 / L_total)))
            N2 = N_non_anomaly - N1
            grid1 = np.linspace(freq[0], anomaly_start, N1, endpoint=False) if N1 > 0 else np.array([])
            grid2 = np.linspace(anomaly_end, freq[-1], N2, endpoint=True) if N2 > 0 else np.array([])
            non_anomaly_grid = np.concatenate((grid1, grid2))
            # Kombiniere beide Raster und sortiere sie
            new_freq = np.sort(np.concatenate((non_anomaly_grid, anomaly_grid)))
        
        # Stelle sicher, dass new_freq genau N_total Punkte hat
        if len(new_freq) != N_total:
            new_freq = np.linspace(new_freq[0], new_freq[-1], N_total)
        
        return new_freq



    @QtCore.Slot()
    def start_odmr_scan(self):
        with self._threadlock:
            if self.module_state() != 'idle':
                self.log.error('Can not start ODMR scan. Measurement is already running.')
                self.sigScanStateUpdated.emit(True)
                return

            microwave = self._microwave()
            sampler = self._data_scanner()

            self.toggle_cw_output(False)
            self.module_state.lock()

            try:
                sample_rate = self._oversampling_factor * self._data_rate
                if self._default_scan_mode != SamplingOutputMode.JUMP_LIST and len(self._scan_frequency_ranges) > 1:
                    mode = SamplingOutputMode.JUMP_LIST
                    self.log.info('Multiple ODMR scan ranges set up. Trying to switch scanner to output mode "JUMP_LIST".')
                else:
                    mode = self._default_scan_mode
                if mode == SamplingOutputMode.JUMP_LIST:
                    frequencies = np.concatenate(self._frequency_data)
                    if self._oversampling_factor > 1:
                        frequencies = np.repeat(frequencies, self._oversampling_factor)
                    samples = len(frequencies)
                elif mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                    frequencies = self._scan_frequency_ranges[0]
                    samples = frequencies[-1]

                sampler.set_sample_rate(sample_rate)
                sampler.set_frame_size(samples)
                microwave.configure_scan(self._scan_power, frequencies, mode, sample_rate)
                microwave.start_scan()
            except:
                self.module_state.unlock()
                self.log.exception('Unable to start ODMR scan. Error while setting up hardware:')
                self.sigScanStateUpdated.emit(False)
                return

            self.clear_all_fits()
            self._elapsed_sweeps = 0
            self._elapsed_time = 0.0
            self.sigElapsedUpdated.emit(self._elapsed_time, self._elapsed_sweeps)
            # Nur im ersten Scan wird _initialize_odmr_data() aufgerufen,
            # damit im zweiten Scan das verfeinerte Frequenzraster erhalten bleibt.
            if not self._is_second_scan:
                self._initialize_odmr_data()
            self.sigScanDataUpdated.emit()
            self.sigScanStateUpdated.emit(True)
            self._start_time = time.time()
            self._sigNextLine.emit()


    def clear_all_fits(self):
        for channel, range_data in self._raw_data.items():
            for range_index, _ in enumerate(range_data):
                self._fit_results[channel][range_index] = None
                self.sigFitUpdated.emit(self._fit_results[channel][range_index], channel, range_index)

    @QtCore.Slot()
    def continue_odmr_scan(self):
        """ Continue ODMR scan.

        @return int: error code (0:OK, -1:error)
        """
        with self._threadlock:
            if self.module_state() == 'locked':
                self.log.error('Can not continue ODMR scan. Measurement is already running.')
                self.sigScanStateUpdated.emit(True)
                return

            # ToDo: see start_odmr_scan
            self.module_state.lock()

            self.sigScanStateUpdated.emit(True)
            self._start_time = time.time() - self._elapsed_time
            self._sigNextLine.emit()



    @QtCore.Slot()
    def clear_odmr_data(self):
        """ Clear the current ODMR data and reset elapsed time/sweeps """
        with self._threadlock:
            if self.module_state() == 'locked':
                self._elapsed_time = 0.0
                self._elapsed_sweeps = 0
                self._initialize_odmr_data()
                self.sigElapsedUpdated.emit(self._elapsed_time, self._elapsed_sweeps)
                self.sigScanDataUpdated.emit()
                self._start_time = time.time()

    @QtCore.Slot()
    def _scan_odmr_line(self):
        """ Perform a single scan over the specified frequency range
        """
        with self._threadlock:
            # If the odmr measurement is not running do nothing and break the Qt signal loop
            if self.module_state() != 'locked':
                return

            try:
                scanner = self._data_scanner()
                new_counts = scanner.acquire_frame()
                if self._oversampling_factor > 1:
                    for ch in new_counts:
                        new_counts[ch] = np.mean(
                            new_counts[ch].reshape(-1, self._oversampling_factor),
                            axis=1
                        )
                self._microwave().reset_scan()
            except:
                self.log.exception('Error while trying to read ODMR scan data from hardware:')
                self.stop_odmr_scan()
                return

            # Add new count data to raw_data array and append if array is too small
            current_line_buffer_size = next(iter(self._raw_data.values()))[0].shape[1]
            if self._elapsed_sweeps == current_line_buffer_size:
                self.log.debug(f'extending data grid for sweep number {self._elapsed_sweeps}')
                expand_arrays = tuple(np.full((r[-1], self.__estimated_lines), np.nan) for r in
                                      self._scan_frequency_ranges)
                self._raw_data = {
                    ch: [np.concatenate((r, expand_arrays[ii]), axis=0) for ii, r in
                         enumerate(range_list)] for ch, range_list in self._raw_data.items()
                }
                self.log.warning(
                    'raw data scan line buffer was not big enough for the entire measurement. '
                    'Buffer will be expanded.\nOld line buffer size was {0:d}, new line buffer '
                    'size is {1:d}.'.format(current_line_buffer_size,
                                            current_line_buffer_size + self.__estimated_lines)
                )

            # shift data in the array "up" and add new data at the "bottom"
            for ch, range_list in self._raw_data.items():
                start = 0
                for range_index, range_params in enumerate(self._scan_frequency_ranges):
                    range_list[range_index] = np.roll(range_list[range_index], 1, axis=1)
                    tmp = new_counts[ch][start:start + range_params[-1]]
                    range_list[range_index][0:len(tmp), 0] = tmp
                    start += range_params[-1]

            # Calculate averaged signal
            self._calculate_signal_data()

            # Update elapsed time/sweeps
            self._elapsed_sweeps += 1
            self._elapsed_time = time.time() - self._start_time

            # Fire update signals
            self.sigElapsedUpdated.emit(self._elapsed_time, self._elapsed_sweeps)
            self.sigScanDataUpdated.emit()
            if self._elapsed_time >= self._run_time:
                self.stop_odmr_scan()
            else:
                self._sigNextLine.emit()
            return

    @QtCore.Slot(str, str, int)
    def do_fit(self, fit_config, channel, range_index):
        """
        Execute the currently configured fit on the measurement data. Optionally on passed data
        """
        if fit_config != 'No Fit' and fit_config not in self._fit_config_model.configuration_names:
            self.log.error(f'Unknown fit configuration "{fit_config}" encountered.')
            return

        x_data = self._frequency_data[range_index]
        y_data = self._signal_data[channel][range_index]

        try:
            fit_config, fit_result = self._fit_container.fit_data(fit_config, x_data, y_data)
        except:
            self.log.exception('Data fitting failed:')
            return

        if fit_result is not None:
            self._fit_results[channel][range_index] = (fit_config, fit_result)
        else:
            self._fit_results[channel][range_index] = None
        self.sigFitUpdated.emit(self._fit_results[channel][range_index], channel, range_index)

    def _get_metadata(self):
        metadata = {'Microwave CW Power (dBm)': self._cw_power,
                    'Microwave Scan Power (dBm)': self._scan_power,
                    'Approx. Run Time (s)': self._elapsed_time,
                    'Number of Frequency Sweeps (#)': self._elapsed_sweeps,
                    'Start Frequencies (Hz)': tuple(rng[0] for rng in self._scan_frequency_ranges),
                    'Stop Frequencies (Hz)': tuple(rng[1] for rng in self._scan_frequency_ranges),
                    'Step sizes (Hz)': tuple(rng[2] for rng in self._scan_frequency_ranges),
                    'Data Rate (Hz)': self._data_rate,
                    'Oversampling factor (Hz)': self._oversampling_factor,
                    'Channel Name': ''}
        for fit_channel in self._fit_results:
            for ii, fit_result in enumerate(self._fit_results[fit_channel]):
                if fit_result:
                    export_dict = FitContainer.dict_result(fit_result[1])
                    metadata[f'fit result (channel "{fit_channel}" range {ii})'] = export_dict
        return metadata

    def _get_raw_column_headers(self, data_channel):
        channel_unit = self.data_constraints.channel_units[data_channel]
        return 'Frequency (Hz)', f'Scan Data ({channel_unit})'

    def _get_signal_column_headers(self):
        channel_units = self.data_constraints.channel_units
        column_headers = ['Frequency (Hz)']
        column_headers.extend(f'{ch} ({channel_units[ch]})' for ch in self._signal_data)
        return tuple(column_headers)

    def _join_channel_raw_data(self, channel):
        """ join raw data for one channel with corresponding frequency data into a single numpy
        array for saving.

        @param str channel: The channel name for which to join the raw data
        """
        channel_data = self._raw_data[channel]
        # Filter raw data to get rid of invalid values (nan or inf)
        joined_data = np.concatenate([raw[:, :self._elapsed_sweeps] for raw in channel_data],
                                     axis=0)
        # add frequency data as first column
        return np.column_stack((np.concatenate(self._frequency_data), joined_data))

    def _join_signal_data(self):
        """ Join and return signal data from all scan ranges into a single numpy array for saving
        """
        joined_data = [np.concatenate(signal) for signal in self._signal_data.values()]
        # add frequency data
        joined_data.insert(0, np.concatenate(self._frequency_data))
        # Join everything in one big array
        return np.column_stack(joined_data)

    @QtCore.Slot(str)
    def save_odmr_data(self, tag=None):
        """ Saves the current ODMR data to a file."""
        with self._threadlock:
            # Create and configure storage helper instance
            timestamp = datetime.datetime.now()
            metadata = self._get_metadata()
            tag = tag + '_' if tag else ''

            # Save raw data in a separate file per data channel
            data_storage = TextDataStorage(root_dir=self.module_default_data_dir,
                                           column_formats='.15e')
            for channel, range_data in self._raw_data.items():
                metadata['Channel Name'] = channel
                column_headers = self._get_raw_column_headers(channel)
                nametag = f'{tag}ODMR_{channel}_raw'
                data = self._join_channel_raw_data(channel)

                # Save raw data for channel
                file_path, _, _ = data_storage.save_data(data,
                                                         metadata=metadata,
                                                         nametag=nametag,
                                                         timestamp=timestamp,
                                                         column_headers=column_headers,
                                                         column_dtypes=float)

                # Save plot images if required. This takes by far the most time to complete.
                if self._save_thumbnails:
                    fig_path_stump = file_path.rsplit('_raw.', 1)[0] + '_range'
                    for range_index, _ in enumerate(range_data):
                        fig = self._draw_figure(channel, range_index)
                        fig_path = f'{fig_path_stump}{range_index:d}'
                        data_storage.save_thumbnail(fig, file_path=fig_path)

            # Save signal data in a single file for all data channels
            del metadata['Channel Name']
            metadata['Averaged Scans (#)'] = self._scans_to_average
            column_headers = self._get_signal_column_headers()
            nametag = f'{tag}ODMR_signal'
            data = self._join_signal_data()

            # Save signal data
            data_storage.save_data(data,
                                   metadata=metadata,
                                   nametag=nametag,
                                   timestamp=timestamp,
                                   column_headers=column_headers,
                                   column_dtypes=[float] * len(column_headers))

    def _draw_figure(self, channel, range_index):
        """ Draw the summary figure to save with the data.

        @param str channel: The data channel name to plot data for.
        @param int range_index: The index for chosen channel data scan range

        @return matplotlib.figure.Figure: a matplotlib figure object to be saved to file.
        """
        freq_data = self._frequency_data[range_index]
        signal_data = self._signal_data[channel][range_index]
        raw_data = self._raw_data[channel][range_index][:, :self._elapsed_sweeps]
        fit_result = self._fit_results[channel][range_index]
        if fit_result is not None:
            fit_x, fit_y = fit_result[1].high_res_best_fit
        unit = self.data_constraints.channel_units[channel]

        # Determine SI unit scaling for signal
        scaled = ScaledFloat(np.max(signal_data))
        signal_unit_prefix = scaled.scale
        if signal_unit_prefix:
            signal_data = signal_data / scaled.scale_val
            if fit_result is not None:
                fit_y = fit_y / scaled.scale_val

        # Determine SI unit scaling for frequency axis
        scaled = ScaledFloat(np.max(freq_data))
        freq_unit_prefix = scaled.scale
        if freq_unit_prefix:
            freq_data = freq_data / scaled.scale_val
            if fit_result is not None:
                fit_x = fit_x / scaled.scale_val

        # Determine SI unit scaling for raw data
        scaled = ScaledFloat(np.max(raw_data))
        raw_unit_prefix = scaled.scale
        if raw_unit_prefix:
            raw_data = raw_data / scaled.scale_val

        # Create figure
        fig, (ax_signal, ax_raw) = plt.subplots(nrows=2, ncols=1)

        # plot signal data
        ax_signal.plot(freq_data, signal_data, linestyle=':', linewidth=0.5, marker='o')
        # Include fit curve if there is one
        if fit_result is not None:
            ax_signal.plot(fit_x, fit_y, marker='None')
        ax_signal.set_ylabel(f'{channel} ({signal_unit_prefix}{unit})')
        ax_signal.set_xlim(min(freq_data), max(freq_data))

        # plot raw data
        raw_data_plot = ax_raw.imshow(raw_data.transpose(),
                                      cmap=plt.get_cmap('inferno'),
                                      origin='lower',
                                      vmin=np.min(raw_data),
                                      vmax=np.max(raw_data),
                                      extent=[min(freq_data),
                                              max(freq_data),
                                              -0.5,
                                              raw_data.shape[1] - 0.5],
                                      aspect='auto',
                                      interpolation='nearest')
        ax_raw.set_xlabel(f'Frequency ({freq_unit_prefix}Hz)')
        ax_raw.set_ylabel('Scan Index')

        # Adjust subplots to make room for colorbar
        fig.subplots_adjust(right=0.8)
        # Add colorbar axis to figure
        colorbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        # Draw colorbar
        colorbar = fig.colorbar(raw_data_plot, cax=colorbar_ax)
        colorbar.set_label(f'{channel} ({signal_unit_prefix}{unit})')
        # remove ticks from colorbar for cleaner image
        colorbar.ax.tick_params(which=u'both', length=0)
        # If we have percentile information, draw that to the figure
        # if percentile_range is not None:
        #     colorbar.ax.annotate(str(percentile_range[0]),
        #                          xy=(-0.3, 0.0),
        #                          xycoords='axes fraction',
        #                          horizontalalignment='right',
        #                          verticalalignment='center',
        #                          rotation=90)
        #     colorbar.ax.annotate(str(percentile_range[1]),
        #                          xy=(-0.3, 1.0),
        #                          xycoords='axes fraction',
        #                          horizontalalignment='right',
        #                          verticalalignment='center',
        #                          rotation=90)
        #     colorbar.ax.annotate('(percentile)',
        #                          xy=(-0.3, 0.5),
        #                          xycoords='axes fraction',
        #                          horizontalalignment='right',
        #                          verticalalignment='center',
        #                          rotation=90)
        return fig