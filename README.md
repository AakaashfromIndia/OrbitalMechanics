# Orbital Mechanics Simulation Suite

A collection of interactive Python simulations for visualizing fundamental orbital mechanics concepts, including two-body dynamics, Hohmann transfers, and bielliptic maneuvers.

![Kepler's law](https://github.com/user-attachments/assets/38dc6640-2ad0-4092-84be-e9bb03907674)

## Overview

This repository contains three comprehensive orbital mechanics simulations built with Python, NumPy, SciPy, and Matplotlib. Each simulation provides real-time visualization of spacecraft trajectories, orbital parameters, and mission events with interactive controls for exploring different scenarios.

## Simulations

### 1. Two Body System (`Two body system.py`)

An interactive simulation of gravitational dynamics between two celestial bodies orbiting their common barycenter. Also follows all three of Kepler's laws

![Kepler's law](https://github.com/user-attachments/assets/38dc6640-2ad0-4092-84be-e9bb03907674)

**Controls:**
- **Mass 1 and 2**: Adjust the masses of the two bodies
- **Semi-major Axis**: Control the size of the orbit
- **Eccentricity**: Modify orbital shape (0 = circular, between 0 and 1 =  elongated)
- **Animation Speed**: Control simulation playback speed



### 2. Hohmann Transfer (`Hohmann Transfer.py`)

A detailed simulation of the classic Hohmann transfer orbit - the most fuel-efficient way to transfer between two circular orbits.

![Hohmann transfer- Made with Clipchamp](https://github.com/user-attachments/assets/6cf1c7b2-555f-41d6-b733-65b6de095d0a)

**Mission Phases:**
1. **Initial Burn**: Spacecraft accelerates from inner circular orbit
2. **Transfer Coast**: Coasting along elliptical transfer orbit
3. **Circularization Burn**: Final burn to achieve outer circular orbit


### 3. Bielliptic Transfer (`Bielliptic-Maneuver.py`)

An advanced simulation of the bielliptic transfer maneuver, which can be more fuel-efficient than Hohmann transfers for large orbit changes.

![Bielliptic maneuver - Made with Clipchamp](https://github.com/user-attachments/assets/e2faf2a8-bec1-4b69-839b-3ceec05effdd)

## Libaries needed

```python
numpy
scipy
matplotlib
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AakaashfromIndia/OrbitalMechanics.git
cd orbital-mechanics-simulations
```

2. Install required dependencies:
```bash
pip install numpy scipy matplotlib
```

3. Run any simulation:
```bash
python Two-body-system.py
python Hohmann-Transfer.py
python Bielliptic-Maneuver.py
```
