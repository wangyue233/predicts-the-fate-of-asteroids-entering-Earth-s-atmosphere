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
            raise NotImplementedError

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100000, dt=0.05, radians=False):
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
        Result = self.solve_atmospheric_entry(radius, velocity, density, strength, angle, init_altitude, dt, radians=False)
        Result = self.calculate_energy(Result)
        outcome = self.analyse_outcome(Result)
        
        return Result, outcome

    def solve_atmospheric_entry(
            self, radius, velocity,
            density, strength, angle, init_altitude=100000, dt=0.05, radians=False):
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

        # calculate initial mass
        mass = density*4/3*np.pi*radius**3
        
        # initialise
        velocity = np.array([velocity])
        mass = np.array([mass])
        altitude = np.array([init_altitude])
        radius = np.array([radius])
        distance = np.zeros((1))

        if radians is True:
            angle = np.array([angle])
        if radians is not True:
            angle = np.array([np.radians(angle)])

        # intermediate step array for RK4
        mas_int = np.zeros((4))
        vel_int = np.zeros((4))
        ang_int = np.zeros((4))
        rad_int = np.zeros((4))
        dis_int = np.zeros((4))
        rho_int = np.zeros((4))
        alt_int = np.zeros((4))
        area_int = np.zeros((4))
        int_step = np.array([1, 2, 2, 1])

        # iteration
        while altitude[-1] > 0:
            rad_int = np.zeros((4))
            
            # first intermediate steps
            rho_int[0] = self.rhoa(altitude[-1])
            area_int[0] = np.pi*radius[-1]**2
            if (self.rhoa(altitude[-1])*velocity[-1]**2 < strength):
                rad_int[0] = 0
            else:
                rad_int[0] = dt*self.radi(rho_int[0], velocity[-1], density)

            mas_int[0] = dt*self.masss(rho_int[0], area_int[0], velocity[-1])
            vel_int[0] = dt*self.velo(rho_int[0], area_int[0], velocity[-1], mass[-1], angle[-1])
            ang_int[0] = dt*self.angl(angle[-1], velocity[-1], rho_int[0], area_int[0], mass[-1], altitude[-1])
            dis_int[0] = dt*self.dist(velocity[-1], angle[-1], altitude[-1])
            alt_int[0] = dt*self.alti(velocity[-1], angle[-1])

            # next three intermediate steps
            for i in range(1, 4):
                temp_mas = (mass[-1] + mas_int[i-1]/int_step[i])
                temp_vel = (velocity[-1] + vel_int[i-1]/int_step[i])
                temp_ang = (angle[-1] + ang_int[i-1]/int_step[i])
                temp_alt = (altitude[-1] + alt_int[i-1]/int_step[i])
                temp_rho = self.rhoa(temp_alt)
        
                if (rad_int[0] == 0):
                    rad_int[i] = 0
                    temp_rad = radius[-1]
                else:
                    temp_rad = (radius[-1] + rad_int[i-1]/int_step[i])
                    rad_int[i] = dt*self.radi(temp_rho, temp_vel, density)
                temp_area = np.pi*temp_rad**2

                mas_int[i] = dt*self.masss(temp_rho, temp_area, temp_vel)
                vel_int[i] = dt*self.velo(temp_rho, temp_area, temp_vel, temp_mas, temp_ang)
                ang_int[i] = dt*self.angl(temp_ang, temp_vel, temp_rho, temp_area, temp_mas, temp_alt)
                dis_int[i] = dt*self.dist(temp_vel, temp_ang, temp_alt)
                alt_int[i] = dt*self.alti(temp_vel, temp_ang)


            # n+1 steps
            alt = altitude[-1] + 1/6*(np.sum(int_step*alt_int))
            mas = mass[-1] + 1/6*(np.sum(int_step*mas_int))
            ang = angle[-1] + 1/6*(np.sum(int_step*ang_int))
            vel = velocity[-1] + 1/6*(np.sum(int_step*vel_int))
            dis = distance[-1] + 1/6*(np.sum(int_step*dis_int))
            rad = radius[-1] + 1/6*(np.sum(int_step*rad_int))
            
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
            distance[-1] = distance[-2] + dt*((velocity[-2]*np.cos(angle[-2]))/(1 + altitude[-2]/self.Rp))
            mass[-1] = mass[-2] + dt*((-self.Ch*rho_n*area_n*velocity[-2]**3)/2/self.Q)
            angle[-1] = angle[-2] + dt*((self.g*np.cos(angle[-2])/velocity[-2]) - \
                (self.Cl*rho_n*area_n*velocity[-2]/2/mass[-2]) - (velocity[-2]*np.cos(angle[-2])/(self.Rp + altitude[-2])))
            velocity[-1] = velocity[-2] + dt*((-self.Cd*rho_n*area_n*velocity[-2]**2)/2/mass[-2] + self.g*np.sin(angle[-2]))

        # change variables for negative mass
        if (mass[-1]<0):
            # mass boundary
            mass[-1] = 0
            # airburst time
            rho_n = self.rhoa(altitude[-2])
            area_n = np.pi*radius[-2]**2
            dt = 2*mass[-2]**2/self.Ch/area_n/rho_n/(velocity[-2]**3)
            time[-1] = time[-2] + dt            
            # other varibles
            distance[-1] = distance[-2] + dt*((velocity[-2]*np.cos(angle[-2]))/(1 + altitude[-2]/self.Rp))
            angle[-1] = angle[-2] + dt*((self.g*np.cos(angle[-2])/velocity[-2]) - \
                (self.Cl*rho_n*area_n*velocity[-2]/2/mass[-2]) - (velocity[-2]*np.cos(angle[-2])/(self.Rp + altitude[-2])))
            altitude[-1] = altitude[-2] + dt*(-velocity[-2]*np.sin(angle[-2]))
            velocity[-1] = velocity[-2] + dt*((-self.Cd*rho_n*area_n*velocity[-2]**2)/2/mass[-2] + self.g*np.sin(angle[-2]))

        return (pd.DataFrame({'velocity': velocity,
                            'mass': mass,
                            'angle': np.degrees(angle),
                            'altitude': altitude,
                            'distance': distance,
                            'radius': radius,
                            'time': time}, index=range(len(mass))))

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

        return (result)

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
        if result.altitude.iloc[index] > 5000:
            burst_peak_dedz = result.dedz.iloc[index]
            burst_altitude = result.altitude.iloc[index]
            burst_total_ke_lost = -0.5*(result.mass.iloc[index]*result.velocity.iloc[index]**2 -\
                result.mass.iloc[0]*result.velocity.iloc[0]**2)/4184e9
            outcome = {'burst_peak_dedz': burst_peak_dedz,
                    'burst_altitude': burst_altitude,
                    'burst_total_ke_lost': burst_total_ke_lost,
                    'outcome': 'Airburst'}
        # airburst and cratering
        elif result.altitude.iloc[index] <= 5000 and result.altitude.iloc[index] > 0:
            burst_peak_dedz = result.dedz.iloc[index]
            burst_altitude = result.altitude.iloc[index]
            burst_total_ke_lost = -0.5*(result.mass.iloc[index]*result.velocity.iloc[index]**2 -\
                result.mass.iloc[0]*result.velocity.iloc[0]**2)/4184e9
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
        return (outcome)
		
    def mars_atmo(self, z):
        pressure = 0.699 * np.exp(-0.00009*z)
        temp_1 = 249.7 - 0.00222*z
        temp_2 = 242.1 - 0.000998*z
        if (z >= 7000):
            temp = temp_1
        else:
            temp = temp_2
        return pressure / (0.1921*temp)
		
    def find_density_tabular(self, alt, atmos):
        ans_index = atmos.Altitude.values.searchsorted(alt) - 1
        density = atmos.iloc[ans_index]['Density'] * np.exp((atmos.iloc[ans_index]['Altitude'] - alt)/atmos.iloc[ans_index]['Height'])
        return density

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
