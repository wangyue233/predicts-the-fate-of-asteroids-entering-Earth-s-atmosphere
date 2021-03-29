from collections import OrderedDict
from pytest import fixture
import pandas as pd
import numpy as np
import math
import os

# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly

@fixture(scope='module')
def armageddon():
    """Perform the module import"""
    import armageddon
    return armageddon

@fixture(scope='module')
def planet(armageddon):
    """Return a default planet with a constant atmosphere"""
    return armageddon.Planet(atmos_func='constant')

@fixture(scope='module')
def tabular(armageddon):
    """Return a defalut planet with a tabulated terrestrial atmosphere"""
    return armageddon.Planet(atmos_func = 'tabular', atmos_filename= 'data/AltitudeDensityTable.csv')

@fixture(scope='module')
def mars(armageddon):
    """Return a defalut planet with the parameters of mars"""
    return armageddon.Planet(atmos_func = 'mars')

@fixture(scope='module')
def input_data():
    input_data = {'radius': 1.,
                  'velocity': 1.0e5,
                  'density': 3000.,
                  'strength': 1e32,
                  'angle': 30.0,
                  'init_altitude':100e3,
                  'dt': 0.05,
                  'radians': False,
                  'ensemble': True
                 }
    return input_data

@fixture(scope='module')
def result(planet, input_data):
    """Solve a default impact for the default planet"""

    result = planet.solve_atmospheric_entry(**input_data)

    return result

def test_import(armageddon):
    """Check package imports"""
    assert armageddon

def test_planet_signature(armageddon):
    """Check planet accepts specified inputs"""
    inputs = OrderedDict(atmos_func='constant',
                         atmos_filename=None,
                         Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
                         alpha=0.3, Rp=6371e3,
                         g=9.81, H=8000., rho0=1.2)

    # call by keyword
    planet = armageddon.Planet(**inputs)

    # call by position
    planet = armageddon.Planet(*inputs.values())

def test_attributes(planet):
    """Check planet has specified attributes."""
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)

def test_solve_atmospheric_entry(result, input_data):
    """Check atmospheric entry solve. 

    Currently only the output type for zero timesteps."""
    
    assert type(result) is pd.DataFrame
    
    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns
    assert result.velocity.iloc[0] == input_data['velocity']
    assert result.altitude.iloc[0] == input_data['init_altitude']
    assert result.distance.iloc[0] == 0.0
    assert result.radius.iloc[0] == input_data['radius']
    assert result.time.iloc[0] == 0.0

def test_calculate_energy(planet, result):
    """Check calculated energy has specified attributes."""
    energy = planet.calculate_energy(result=result)

    print(energy)

    assert type(energy) is pd.DataFrame
    
    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns

def test_analyse_outcome(planet, result):
    """Check the type of analyse outcome."""
    outcome = planet.analyse_outcome(result)

    assert type(outcome) is dict

def test_ensemble(planet, armageddon):
    """Check ensemble function gives the value of burst_altitude."""
    fiducial_impact = {'radius': 10.0,
                       'angle': 45.0,
                       'strength': 1e5,
                       'velocity': 20e3,
                       'density': 3000}
    
    ensemble = armageddon.ensemble.solve_ensemble(planet,
                                                  fiducial_impact,
                                                  variables=[], radians=False,
                                                  rmin=8, rmax=12)

    assert 'burst_altitude' in ensemble.columns

def test_mars_atmo(mars):
    """Check mars_atmo function returns the correct atmospheric density."""
    temp_1 = 8000
    temp_2 = 6000
    pressure_1 = 0.699 * np.exp(-0.00009*temp_1)
    pressure_2 = 0.699 * np.exp(-0.00009*temp_2)

    assert mars.mars_atmo(temp_1) == pressure_1 / (0.1921*(249.7 - 0.00222 * temp_1))
    assert mars.mars_atmo(temp_2) == pressure_2 / (0.1921*(242.1 - 0.000998* temp_2))

def test_find_density_tabular(tabular):
    """Check tabular function returns the correct atmospheric density."""
    altitude = 555
    columns = ["Altitude", "Density", "Height"]
    atmos = pd.read_csv('data/AltitudeDensityTable.csv', names=columns, skiprows=6, sep = " ")
    density = tabular.find_density_tabular(altitude, atmos)

    assert density == 1.16161 * np.exp(-5/10274.7337022231)
     
def test_analytical_solution(planet, input_data, armageddon):
    """Compute analytical solution and compare it with numerical solution. """
    # Input constants
    Cd = 1
    H = 8000
    rho0 = 1.2
    A = np.pi*input_data["radius"]**2
    m = 4/3*np.pi*input_data["radius"]**3*input_data["density"]
    K = Cd*A*rho0/(2*m)

    # Set the condition for analytical solution
    another_planet = armageddon.Planet(Cd=1., Ch=0, Q=1e7, Cl=0, alpha=0.3, Rp=math.inf,
                 g=0, H=8000., rho0=1.2)
    
    # Calculate the analytical solution and compare it with numerical solution
    another_result,another_outcome = another_planet.impact(**input_data)
    analytical_velocity = input_data["velocity"] * np.exp(H*K/np.sin(input_data["angle"]/180*np.pi)\
                   *(np.exp(- input_data["init_altitude"]/H)-np.exp(-another_result['altitude']/H)))
    rms = sum((analytical_velocity-another_result['velocity'])**2)
    rms = math.sqrt(rms/len(analytical_velocity))

    assert abs(rms)/input_data["velocity"] < 0.3
    
def test_find_parameter(armageddon):
    """Compute estimated parameters and compare it with observed value. """
    asteroid = armageddon.Planet()
    e_radius, e_strength = asteroid.find_parameter(filename=os.getcwd()+'/data/ChelyabinskEnergyAltitude.csv')
    result, outcome = asteroid.impact(
    radius=e_radius, angle=18.3, strength=e_strength, velocity=19.2e3, density=3300,init_altitude=100e3)

    energy = pd.read_csv(os.getcwd()+'/data/ChelyabinskEnergyAltitude.csv')
    burst_attitude = energy['Height (km)'].iloc[energy['Energy Per Unit Length (kt Km^-1)'].idxmax()]
    burst_energy = max(energy['Energy Per Unit Length (kt Km^-1)'])

    assert np.abs(outcome['burst_altitude']/burst_attitude/1000-1) < 0.2
    assert np.abs(outcome['burst_peak_dedz']/burst_energy-1) < 0.2
