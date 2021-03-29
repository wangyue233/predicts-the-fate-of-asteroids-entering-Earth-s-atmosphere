# Perform imports
import numpy as np
import pandas as pd
from . import solver
from scipy.special import erf, erfinv
import dask
from dask import delayed

def solve_ensemble(
        planet,
        fiducial_impact,
        variables,
        radians=False,
        rmin=8, rmax=12,
        samples=30):
    """
    Run asteroid simulation for a distribution of initial conditions and
    find the burst distribution

    Parameters
    ----------

    planet : object
        The Planet class instance on which to perform the ensemble calculation

    fiducial_impact : dict
        Dictionary of the fiducial values of radius, angle, strength, velocity
        and density

    variables : list
        List of strings of all impact parameters to be varied in the ensemble
        calculation

    rmin : float, optional
        Minimum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    rmax : float, optional
        Maximum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    Returns
    -------

    ensemble : DataFrame
        DataFrame with columns of any parameters that are varied and the
        airburst altitude
    """
    
    # density needs a special probability array because certain values cause it to be infinite
    probabilities = np.linspace(0,1,samples)
    probabilities_den = np.linspace(0.01,0.99,samples)
	
    # start by setting defaults
    radius = np.full(samples,fiducial_impact['radius'])
    angle = np.full(samples,fiducial_impact['angle'])
    strength = np.full(samples,fiducial_impact['strength'])
    velocity = np.full(samples,fiducial_impact['velocity'])
    density = np.full(samples,fiducial_impact['density'])
	
    # generate the random versions of all variables
    # (could save some time by only generating them when needed, but this is very fast even for large sample sizes)
    var_rad = probabilities * (rmax - rmin) + rmin
    var_ang = np.arccos(np.sqrt(1-probabilities))
    var_ang = var_ang * 180 / np.pi
    var_ang[var_ang < 10] = 10
    var_str = np.power(10,probabilities*np.log10(1e4)+np.log10(1e3))
    
    erf = 2*probabilities_den - 1
    middle = erfinv(erf)
    var_den = middle*np.sqrt(2)*1000+3000
    var_den[var_den < 1] = 1

    # will be filled with which variables are varying, so that only those are returned
    columns = []

    # replace constant values with variable ones if they are in "variables"
    for var in variables:
        if var == 'radius' or radius.any() <= 0:
            np.random.shuffle(var_rad)
            radius = var_rad
            columns.append(radius)
        if var == 'angle' or angle.any() <= 0:
            np.random.shuffle(var_ang)
            angle = var_ang
            columns.append(angle)
        if var == 'strength' or strength.any() <= 0:
            np.random.shuffle(var_str)
            strength = var_str
            columns.append(strength)
        if var == 'velocity' or velocity.any() <= 0:
            inf_velocity = np.array([inverse_F(u, 11) for u in probabilities])
            v_escape = 11.2
            velocity = np.sqrt(v_escape ** 2 + inf_velocity ** 2) * 1e3
            columns.append(velocity)
        if var == 'density' or density.any() <= 0:
            np.random.shuffle(var_den)
            density = var_den
            columns.append(density)

    # generate delayed objects for all outputs
    outcome = []
    for i in range(samples):
        output = delayed(planet.impact)(
                     radius=radius[i], angle=angle[i], strength=strength[i], velocity=velocity[i], density=density[i], init_altitude= 100000, dt = 0.05, ensemble=True)
        outcome.append(output[1])
    
    # then compute them ("outputs" is now a dictionary containing burst altitudes)
    outputs = dask.compute(*outcome, scheduler='processes')

    # now extract the altitudes from the dict
    results = []
    
    for i in range(samples):
        try:
            results.append(outputs[i]['burst_altitude'])
        except KeyError: # if no burst happened, we can assume burst altitude to be 0 (cratering impact)
            results.append(0)

    # finally, build the DataFrame returning burst altitudes and the corresponding values which created them
    distribution = pd.DataFrame()
    for i in range(len(variables)):
        distribution[variables[i]] = columns[i]

    distribution['burst_altitude'] = results
    
    # if "burst altitude" is higher than initial altitude, that actually means the asteroid left earth orbit
    # so do not return that result
    return distribution.loc[distribution['burst_altitude'] < 100000]

def F(x, a):
    """
    The given probability function for velocity at infinity
    
    Parameters
    ----------

    x: float
    velocity to get probability of (actually gets P(<x))
    
    a: int
    constant, 11 km / s

    Returns
    -------

    Probability of velocity being less than x
    """
    return erf(x/(a*np.sqrt(2)))-(x/a)*np.exp(-x**2/2/a**2)*np.sqrt(2/np.pi)

def inverse_F(p, a):
    """
    Inverts the probability function to get velocity for each specific probability
    
    Parameters
    ----------

    p: float
    probability to find x less than
    
    a: int
    constant, 11 km / s

    Returns
    -------

    Velocity v such that P(<v) = p
    """
    # generate a lot of points to be accurate, velocity is set to between 0-50 because it cannot be negative and 50 is a good upper bound
    candidates = np.linspace(0,50,10000)
    for x in candidates:
        if F(x, a) >= p:
            return x
    return 50

if __name__ == '__main__':
    earth = solver.Planet()
    result = solve_ensemble(earth, {'radius': 10, 'angle': 45, 'strength': 1e5, 'velocity': 20e3, 'density': 3000}, variables=['radius','angle','density','velocity','strength'])
    altitudes = result['burst_altitude']
    count, bins, ignored = plt.hist(altitudes, 40, facecolor='green') 
    print(result)
