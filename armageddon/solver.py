import numpy as np
import pandas as pd


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------

        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function ``rho = rho0 exp(-z/H)``.
            Options are ``exponential``, ``tabular``, ``constant`` and ``mars``

        atmos_filename : string, optional
            If ``atmos_func`` = ``'tabular'``, then set the filename of the table
            to be read in here.

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        Returns
        -------

        None
        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0

        if atmos_func == 'exponential':
            self.rhoa = lambda z: rho0 * np.exp(-z/H)
        elif atmos_func == 'tabular':
            columns = ["Altitude", "Density", "Height"]
            self.atmos = pd.read_csv(atmos_filename, names=columns, skiprows=6, sep = " ")
            self.rhoa = lambda z: self.find_density_tabular(z, self.atmos)
        elif atmos_func == 'mars':
            self.rhoa = lambda z: self.mars_atmo(z)
        elif atmos_func == 'constant':
            self.rhoa = lambda x: rho0
        else:
            print("Invalid atmos func! Using default instead")
            self.rhoa = lambda z: rho0 * np.exp(-z/H)

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100000, dt=0.05, radians=False, ensemble=False):
        """
        Solve the system of differential equations for a given impact event.
        Also calculates the kinetic energy lost per unit altitude and
        analyses the result to determine the outcome of the impact.

        Parameters
        ----------

        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        Returns
        -------

        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``, ``dedz``

        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.

            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        # call methods
        Result = self.solve_atmospheric_entry(radius, velocity, density, strength, angle, init_altitude, dt, radians=False, ensemble=ensemble)
        Result = self.calculate_energy(Result)
        outcome = self.analyse_outcome(Result)
        
        return Result, outcome

    def solve_atmospheric_entry(
            self, radius, velocity,
            density, strength, angle, init_altitude=100000, dt=0.05, radians=False, ensemble=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------

        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        Returns
        -------
        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``
        """

        # for the output, use given dt. But for other cases, use ours
        out_dt = dt
        dt = 0.5
        keep = int(out_dt/dt)
        
        # if output timestep is smaller than ours, use it instead
        if not keep == out_dt/dt or out_dt<dt:
            dt = out_dt
            keep = 1
            
        # calculate initial mass
        mass = density*4/3*np.pi*radius**3
        
        # initialise
        velocity = np.array([velocity])
        mass = np.array([mass])
        altitude = np.array([init_altitude])
        radius = np.array([radius])
        distance = np.zeros((1))
        
        # always work with radians
        if radians is True:
            angle = np.array([angle])
        if radians is not True:
            angle = np.array([np.radians(angle)])

        # iteration
        while altitude[-1] > 0:
            
            # air density and cross-section area at step n
            rho_n = self.rhoa(altitude[-1])
            area_n = np.pi*radius[-1]**2

            # intermediate steps
            alt_int = altitude[-1] + dt*self.alti(velocity[-1], angle[-1])
            rho_int = self.rhoa(alt_int)
            mas_int = mass[-1] + dt*self.masss(rho_n, area_n, velocity[-1])
            ang_int = angle[-1] + dt*self.angl(angle[-1], velocity[-1], rho_n, area_n, mass[-1], altitude[-1])
            vel_int = velocity[-1] + dt*self.velo(rho_n, area_n, velocity[-1], mass[-1], angle[-1])

            # n+1 step cross-section breakup conditions
            if (self.rhoa(altitude[-1])*velocity[-1]**2 < strength):
                rad_int = radius[-1]
                area_int = np.pi*radius[-1]**2
                rad = radius[-1]
            else:
                rad_int = radius[-1] + dt*self.radi(rho_n, velocity[-1], density)
                area_int = np.pi*rad_int**2
                rad = radius[-1] + 0.5*dt*(self.radi(rho_n, velocity[-1], density) + self.radi(rho_int, vel_int, density))

            # n+1 steps other varibles
            alt = altitude[-1] + 0.5*dt*(self.alti(velocity[-1], angle[-1]) + self.alti(vel_int, ang_int))
            mas = mass[-1] + 0.5*dt*(self.masss(rho_n, area_n, velocity[-1]) + self.masss(rho_int, area_int, vel_int))
            ang = angle[-1] + 0.5*dt*(self.angl(angle[-1], velocity[-1], rho_n, area_n, mass[-1], altitude[-1]) + \
                self.angl(ang_int, vel_int, rho_int, area_int, mas_int, alt_int))
            vel = velocity[-1] + 0.5*dt*(self.velo(rho_n, area_n, velocity[-1], mass[-1], angle[-1]) + \
                self.velo(rho_int, area_int, vel_int, mas_int, ang_int))
            dis = distance[-1] + 0.5*dt*(self.dist(velocity[-1], angle[-1], altitude[-1]) + \
                self.dist(vel_int, ang_int, alt_int))
            
            # append new values into array
            altitude = np.append(altitude, [alt])
            mass = np.append(mass, [mas])
            radius = np.append(radius, [rad])
            angle = np.append(angle, [ang])
            velocity = np.append(velocity, [vel])
            distance = np.append(distance, [dis])
            
            # check for negative mass
            if mass[-1] < 0:
                break
            # asteroid is going away from earth
            if (altitude[-1] > altitude[0] + 100):
                break
        
        time = np.linspace(0, dt*(len(mass)-1), len(mass))

        # impact with ground assuming no break up during the last step
        if (altitude[-1] < 0):
            # ground boundary
            altitude[-1] = 0
            # impact time
            dt = altitude[-2]/velocity[-2]/np.sin(angle[-2])
            time[-1] = time[-2] + dt
            rho_n = self.rhoa(altitude[-2])
            area_n = np.pi*radius[-2]**2
            
            # other variables
            distance[-1] = distance[-2] + dt*(self.dist(velocity[-2], angle[-2], altitude[-2]))
            mass[-1] = mass[-2] + dt*(self.masss(rho_n, area_n, velocity[-2]))
            angle[-1] = angle[-2] + dt*(self.angl(angle[-2], velocity[-2], rho_n, area_n, mass[-2], altitude[-2]))
            velocity[-1] = velocity[-2] + dt*(self.velo(rho_n, area_n, velocity[-2], mass[-2], angle[-2]))

        # change variables for negative mass
        if (mass[-1] < 0):
            # mass boundary
            mass[-1] = 0
            # airburst time
            rho_n = self.rhoa(altitude[-2])
            area_n = np.pi*radius[-2]**2
            dt = 2*mass[-2]**2/self.Ch/area_n/rho_n/(velocity[-2]**3)
            time[-1] = time[-2] + dt            
            # other varibles
            distance[-1] = distance[-2] + dt*(self.dist(velocity[-2], angle[-2], altitude[-2]))
            altitude[-1] = altitude[-2] + dt*(self.alti(velocity[-2], angle[-2]))
            angle[-1] = angle[-2] + dt*(self.angl(angle[-2], velocity[-2], rho_n, area_n, mass[-2], altitude[-2]))
            velocity[-1] = velocity[-2] + dt*(self.velo(rho_n, area_n, velocity[-2], mass[-2], angle[-2]))

        # only keep rows where time is a multiple of dt
        return (pd.DataFrame({'velocity': velocity,
                            'mass': mass,
                            'angle': np.degrees(angle),
                            'altitude': altitude,
                            'distance': distance,
                            'radius': radius,
                            'time': time}, index=range(len(mass))).iloc[::keep,:])

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------

        result : DataFrame
            A pandas DataFrame with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns
        -------

        Result : DataFrame
            Returns the DataFrame with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """

        # initialise
        dedz = np.zeros((len(result.mass)))
        mass = np.array(result.mass.iloc[:])
        velocity = np.array(result.velocity.iloc[:])
        altitude = np.array(result.altitude.iloc[:])
        
        # calculate kinetic energy loss
        dedz[:-1] = (mass[1:]*velocity[1:]**2 - mass[:-1]*velocity[:-1]**2)/2/(altitude[1:] - altitude[:-1])/4184e6
        dedz[-1] = (-mass[-1]*velocity[-1]**2)/2/(-altitude[-2])/4184e6
        result.insert(len(result.columns),
                      'dedz', np.array(dedz))

        return result

    def analyse_outcome(self, result):
        """
        Inspect a prefound solution to calculate the impact and airburst stats

        Parameters
        ----------

        result : DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time

        Returns
        -------

        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.

            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """

        index = result.dedz.idxmax()

        # airburst
        if result.altitude[index] > 5000:
            burst_peak_dedz = result.dedz[index]
            burst_altitude = result.altitude[index]
            burst_total_ke_lost = -0.5*(result.mass[index]*result.velocity[index]**2 -\
                result.mass[0]*result.velocity[0]**2)/4184e9
            outcome = {'burst_peak_dedz': burst_peak_dedz,
                    'burst_altitude': burst_altitude,
                    'burst_total_ke_lost': burst_total_ke_lost,
                    'outcome': 'Airburst'}
        
        # airburst and cratering
        elif result.altitude[index] <= 5000 and result.altitude[index] > 0:
            burst_peak_dedz = result.dedz[index]
            burst_altitude = result.altitude[index]
            burst_total_ke_lost = -0.5*(result.mass[index]*result.velocity[index]**2 -\
                result.mass[0]*result.velocity[0]**2)/4184e9
            impact_time = result.time.iloc[-1]
            impact_mass = result.mass.iloc[-1]
            impact_speed = result.velocity.iloc[-1]
            outcome = {'burst_peak_dedz': burst_peak_dedz,
                    'burst_altitude': burst_altitude,
                    'burst_total_ke_lost': burst_total_ke_lost,
                    'impact_time': impact_time,
                    'impact_mass': impact_mass,
                    'impact_speed': impact_speed,
                    'outcome': 'Airburst and Cratering'}
            
        # cratering
        else:
            impact_time = result.time.iloc[-1]
            impact_mass = result.mass.iloc[-1]
            impact_speed = result.velocity.iloc[-1]

            outcome = {'impact_time': impact_time,
                    'impact_mass': impact_mass,
                    'impact_speed': impact_speed,
                    'outcome': 'Cratering'}

        return outcome

    def mars_atmo(self, z):
        """
        Function to represent the atmosphere of mars
        
        Parameters
        ----------

        z : float
            the altitude at which to find atmosphere density

        Returns
        -------
        float: density of marsian atmosphere at this altitude

        """
        pressure = 0.699 * np.exp(-0.00009*z)
        temp_1 = 249.7 - 0.00222*z
        temp_2 = 242.1 - 0.000998*z
        if (z >= 7000):
            temp = temp_1
        else:
            temp = temp_2
        return pressure / (0.1921*temp)

    def find_density_tabular(self, alt, atmos):
        """
        Function to represent a tabular representation of earth's atmosphere
        
        Parameters
        ----------

        alt : float
            the altitude at which to find atmosphere density
            
        atmos: DataFrame
             Contains density altitude and height values which can be used in interpolation

        Returns
        -------
        density: float
            The density of atmosphere at this altitude on earth
        """
        ans_index = atmos.Altitude.values.searchsorted(alt) - 1
        density = atmos.iloc[ans_index]['Density'] * np.exp((atmos.iloc[ans_index]['Altitude'] - alt)/atmos.iloc[ans_index]['Height'])
        return density

    # Helper functions for RK2
    def masss(self, rho, area, velocity):
        mass_f = ((-self.Ch*rho*area*velocity**3)/2/self.Q)
        return mass_f
    def velo(self, rho, area, velocity, mass, angle):
        velo_f = ((-self.Cd*rho*area*velocity**2)/2/mass + self.g*np.sin(angle))
        return velo_f
    def angl(self, angle, velocity, rho, area, mass, altitude):
        angl_f = ((self.g*np.cos(angle)/velocity) - (self.Cl*rho*area*velocity/2/mass) - \
            (velocity*np.cos(angle)/(self.Rp + altitude)))
        return angl_f
    def radi(self, rho, velocity, density):
        radi_f = np.sqrt(7/2*self.alpha*rho/density)*velocity
        return radi_f
    def dist(self, velocity, angle, altitude):
        dist_f = (((velocity*np.cos(angle))/(1 + altitude/self.Rp)))
        return dist_f
    def alti(self, velocity, angle):
        alti_f = (-velocity*np.sin(angle))
        return alti_f
    
    # Chelyabinsk Inversion (find radius and strength )
   
    def find_parameter(self,filename='../data/ChelyabinskEnergyAltitude.csv',velocity=19.2e3,angle=18.3, \
                   density=3300,radians=False):
        """
        determine asteroid parameters (e.g., strength and radius) that best fit an observed energy deposition curve

        Parameters
        ----------

        filename : string
            the csv file path for the observed energy deposition curve

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        Returns
        -------

        estimated_radius : float
            the estimated radius to best fit the observed value
            
        estimated_strength : float
            the estimated strength to best fit the observed value

        """
        # get the observed energy
        energy = pd.read_csv(filename)
        # observed altitude
        ob_alt = np.array(energy.iloc[:]['Height (km)'])
        # observed energy loss per unit height
        ob_eng = np.array(energy.iloc[:]['Energy Per Unit Length (kt Km^-1)'])
        # observed burst altitude
        burst_attitude = energy['Height (km)'].iloc[energy['Energy Per Unit Length (kt Km^-1)'].idxmax()]
        # observed burst peak energy
        burst_energy = max(energy['Energy Per Unit Length (kt Km^-1)'])
        # minimal estimated strength
        z = ob_alt[0]*1e3
        y = 1.2 * np.exp(-z / 8000) *velocity**2
        # find the estimated value for radius since normally the bigger the radius, the bigger the peak burst energy
        analyed_energy = 0
        max_radius = 0
        for i in range(5,20):
            result, outcome = self.impact(
            radius=i, angle=angle, strength=y, velocity=velocity, density=density, init_altitude=100e3, radians = radians, ensemble=True)
            analyed_energy = outcome['burst_peak_dedz']
            if analyed_energy > burst_energy:
                max_radius = i
                break
        # find the best fit for burst altitde and burst peak energy
        # under the range [max_radius-2,max_radius+1]X[strength,10strength]
        df = pd.DataFrame({'radius': [], 'strength': [],'error_attitude':[],'error_energy':[]})
        for i in range(max_radius-2, max_radius+1):
            for s in  np.linspace(y, y*20, 10):
                result, outcome = self.impact(
            radius=i, angle=18.3, strength=s, velocity=19.2e3, density=3300,init_altitude=100e3,ensemble=True)
                df = df.append({'radius': i, 'strength': s,'error_attitude': \
                                np.abs(outcome['burst_altitude']/burst_attitude/1000-1), \
                               'error_energy':np.abs(outcome['burst_peak_dedz']-burst_energy)}, ignore_index=True)
        df['error_total'] = df['error_attitude']+df['error_energy']
        estimated_radius = df['radius'].iloc[df['error_total'].idxmin()]
        estimated_strength = df['strength'].iloc[df['error_total'].idxmin()]
        # find the best fit for burst altitde and burst peak energy
        # under the range [max_radius-0.5,max_radius+0.5]X[strength,2strength]
        new_df = pd.DataFrame({'radius': [], 'strength': [],'error_attitude':[],'error_energy':[]})
        for i in np.linspace(estimated_radius-0.5, estimated_radius+0.5,10):
            for s in  np.linspace(estimated_strength, estimated_strength*2,10):
                result, outcome = self.impact(
            radius=i, angle=18.3, strength=s, velocity=19.2e3, density=3300,init_altitude=100e3,ensemble=True)
                new_df = new_df.append({'radius': i, 'strength': s,'error_attitude': \
                                np.abs(outcome['burst_altitude']/burst_attitude/1000-1), \
                               'error_energy': np.abs(outcome['burst_peak_dedz']-burst_energy)}, ignore_index=True)
        new_df['error_total'] = new_df['error_attitude'] + new_df['error_energy']
        estimated_radius = new_df['radius'].iloc[new_df['error_total'].idxmin()]
        estimated_strength = new_df['strength'].iloc[new_df['error_total'].idxmin()]
        return estimated_radius,estimated_strength


