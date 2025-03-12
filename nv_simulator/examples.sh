#!/bin/bash

# Beispiele für die Verwendung des NV-Zentrum-Simulators

echo "NV-Zentrum Quantencomputer-Simulator - Beispielskript"
echo "===================================================="
echo

# Sicherstellen, dass das Skript ausführbar ist
chmod +x nv_simulator.py

echo "1. Quantendynamik Simulation"
echo "---------------------------"
echo "Simulation der Quantendynamik eines NV-Zentrums unter Mikrowellenantrieb"
echo "Befehl: ./nv_simulator.py --type dynamics --mw-freq 2.87e9 --mw-ampl 10e6 --t-max 1000 --initial ms0"
echo
./nv_simulator.py --type dynamics --mw-freq 2.87e9 --mw-ampl 10e6 --t-max 1000 --initial ms0
echo 
echo "Simulation mit externem Magnetfeld (20 mT in z-Richtung)"
echo "Befehl: ./nv_simulator.py --type dynamics --Bz 0.02 --mw-freq 2.9e9 --mw-ampl 5e6 --t-max 1000"
echo
./nv_simulator.py --type dynamics --Bz 0.02 --mw-freq 2.9e9 --mw-ampl 5e6 --t-max 1000
echo
echo

echo "2. ODMR-Spektrum Simulation"
echo "-------------------------"
echo "Simulation eines ODMR-Spektrums ohne externes Magnetfeld"
echo "Befehl: ./nv_simulator.py --type odmr --freq-min 2.77e9 --freq-max 2.97e9 --freq-points 101"
echo
./nv_simulator.py --type odmr --freq-min 2.77e9 --freq-max 2.97e9 --freq-points 101
echo
echo "ODMR-Spektrum mit externem Magnetfeld (10 mT in z-Richtung)"
echo "Befehl: ./nv_simulator.py --type odmr --Bz 0.01 --freq-min 2.77e9 --freq-max 2.97e9"
echo
./nv_simulator.py --type odmr --Bz 0.01 --freq-min 2.77e9 --freq-max 2.97e9
echo
echo

echo "3. Rabi-Oszillationen Simulation"
echo "-----------------------------"
echo "Simulation von Rabi-Oszillationen mit variierender Mikrowellenamplitude"
echo "Befehl: ./nv_simulator.py --type rabi --mw-amp-min 0 --mw-amp-max 30e6 --mw-amp-points 31 --pulse-duration 500"
echo
./nv_simulator.py --type rabi --mw-amp-min 0 --mw-amp-max 30e6 --mw-amp-points 31 --pulse-duration 500
echo
echo

echo "Weitere Optionen und Parameter finden Sie mit ./nv_simulator.py --help"