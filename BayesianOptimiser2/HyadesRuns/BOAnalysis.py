"""
Radius  = ncread(file,'R')
Time  = ncread(file,'DumpTimes')
Radiuscm  = ncread(file,'Rcm')
Density  = ncread(file,'Rho') # g/cm^3
ElecTemp  = ncread(file,'Te')*1000 # Converted to eV from keV
IonTemp  = ncread(file,'Ti')*1000 # Converted to eV from keV
Neutrons = sum((0.8/(14.1 * 1.60218e-6))*ncread(file, 'Bpeprd')) # x 0.8 to get fraction of energy in neutrons, then divide by neutron energy in MeV
NeutronRate = sum((0.8/(14.1 * 1.60218e-6))*ncread(file, 'Bpeprdr'))
OldRhoR = Density.*Radius(1:end-1, :) 
Volume =  ncread(file,'Vol') # cm^3
Mass = Volume(2:end,:).*Density # g
Pressure = ncread(file,'Pres').*0.0000001 # Converted to J/cm^3 from dyn/cm^2
PressureGPa = Pressure.*10^-3
TNOutput = ncread(file, 'Bpeprdr') # TN burn rate
TNInput = ncread(file, 'Bpedep').*10^-7 # Deposited TN energy per zone, converted from erg to J
TNInputRate = [zeros(size(TNInput,1),1),diff(TNInput,1,2)] # TN deposition rate
Velocity = ncread(file, 'U')./100000 # Converted to km/s from cm/s
IonThermalEnergy  = ncread(file,'Eion').*10^-7 # Ion Thermal Energy, converted from erg to J.
ElectronThermalEnergy  = ncread(file,'Eelc').*10^-7 # Electron Thermal Energy, converted from erg to J.
# KineticEnergy  = ncread(file,'Ekint').*10^-7
DepositedLaserPowerZones = ncread(file,'Deplas').*10^-7 # Deposited Laser Energy per zone, converted from erg to J.
DepositedLaserPower = sum(DepositedLaserPowerZones).' # Total deposited laser power (over all zones)
DepositedEnergy = trapz(Time, DepositedLaserPower) # Integrate for deposited energy

LaserEnergy  = ncread(file,'Elasin')*(10^-7)/0.8    # A 0.8 factor is applied to input power to account for cross beam energy transfer (i.e., a power E/0.8 is required in real life to achieve E in Hyades). Laser Energy and Laser Power corresponds to real energy, while Simulation Power describes Hyades power.

LaserPower=[0 diff(LaserEnergy)./diff(Time)]
SimulationPower = LaserPower*0.8
"""


import os
import glob
import numpy as np
import netCDF4 as nc
import scipy.constants as con


def ShellParameters(file):
    """ 
    Calculates the inner and outer shell indices and radii at each timestep.
    
    Can be used to calculate any shell-dependent quantities (CR, IFAR etc.). Shell may be poorly defined during certain stages of the implosion, also need to determine when the shell is well defined and only calculate values in this window.

    Parameters:
    file (string): path to file containing Hyades data.
    
    Returns:
    InnerShellIndex (1d array, int): The index of the inner shell at each timestep.
    InnerShellRadius (1d array, float): The radius of the inner shell at each timestep.
    OuterShellIndex (1d array, int): The index of the outer shell at each timestep.
    OuterShellRadius (1d array, float): The radius of the outer shell at each timestep.
    """

    data = nc.Dataset(file)

    Density = np.array(data['Rho'])
    X = np.array(data['R'])

    InnerShellIndex = np.empty(len(Density))
    InnerShellRadius = np.empty(len(Density))

    OuterShellIndex = np.empty(len(Density))
    OuterShellRadius = np.empty(len(Density))

    for timestep, density in enumerate(Density):
        # Find the inner shell radius at all timesteps.
        truth_array = (density - density[0]) > ( (max(density) - density[0]) * 0.135 )

        # Try loop incase central denisty is the largest.
        try: 
            InnerShellThresholdIndex = np.where(truth_array)[0][0]
        except:
            InnerShellThresholdIndex = 0
        
        InnerShellIndex[timestep] = int(InnerShellThresholdIndex)
        InnerShellRadius[timestep] = X[timestep][InnerShellThresholdIndex]

        # Find the outer shell index at all timesteps.
        truth_array = density > (max(density)*0.135)
        OuterShellThresholdIndex = np.where(truth_array)[0][-1]
        OuterShellIndex[timestep] = int(OuterShellThresholdIndex)
        OuterShellRadius[timestep] = X[timestep][OuterShellThresholdIndex]

    # Need to determine when the shell is well defined, otherwise CR's of 'inf' will often be returned. 
    

    return InnerShellIndex, InnerShellRadius, OuterShellIndex, OuterShellRadius


def BangTime(file):
    """
    Defined as the timestep with maximum neutron production.

    Parameters:
    file (string): path to file containing Hyades data.

    Returns:
    bang_time_index (int): Bang time index
    bang_time (float): Bang time
    """

    data = nc.Dataset(file)
    Times = np.array(data['DumpTimes'])

    # Find the number of neutrons released per timestep. 1 erg = 1e-7 J, 1 J = 6.242e+12 MeV.
    NeutronRate = np.empty(len(data['Bpeprd']))
    for timestep in range(len(data['Bpeprd'])):
        if timestep == 0: NeutronRate[timestep] = sum(data['Bpeprd'][timestep])
        else: NeutronRate[timestep] = (0.8/(14.1 * 1.60218e-6)) * ( sum(data['Bpeprd'][timestep]) - sum(data['Bpeprd'][timestep - 1]) )

    bang_time_index = np.argmax(NeutronRate)
    bang_time = Times[bang_time_index]

    return bang_time_index, bang_time


def BurnWidth(file): 
    """
    Find the maximum neutron production rate. Find when this rate falls to half either side. Burn width is this duration.

    Parameters:
    file (string): path to file containing Hyades data.

    Returns:
    burn_width (float): Duration when neutron production is >0.5*max(NeutronYeild)
    """

    data = nc.Dataset(file)
    Times = np.array(data['DumpTimes'])

    # Find the number of neutrons released per timestep. 1 erg = 1e-7 J, 1 J = 6.242e+12 MeV.
    NeutronRate = np.empty(len(data['Bpeprd']))
    for timestep in range(len(data['Bpeprd'])):
        if timestep == 0: NeutronRate[timestep] = sum(data['Bpeprd'][timestep])
        else: NeutronRate[timestep] = (0.8/(14.1 * 1.60218e-6)) * ( sum(data['Bpeprd'][timestep]) - sum(data['Bpeprd'][timestep - 1]) )

    upper_burn_index = np.where(NeutronRate > 0.5 * np.max(NeutronRate))[0][-1]
    lower_burn_index = np.where(NeutronRate > 0.5 * np.max(NeutronRate))[0][0]

    upper_burn_time = Times[upper_burn_index]
    lower_burn_time = Times[lower_burn_index]

    burn_width = upper_burn_time - lower_burn_time

    return burn_width


def Gain(file, CBET_MULTIPLIER=0.8, AH_energy_J=0, AH_efficiency=0.094):
    """ 
    calculates the 'total' gain and 'neutron' gain of the implosion. 
    
    Parameters:
    file (string): path to file containing Hyades data.
    CBET_MULTIPLIER (0 < float <= 1): Fraction of laser energy remaining after CBET.
    
    Returns:
    TotalGain (float): Gain calculated from the total energy vs required laser energy. 
    NeutronGain (float): Gain calculated from the number of neutrons generated. 
    """

    data = nc.Dataset(file)

    # Find the number of neutrons released throughout the simulation (integrated up to the current timestep). 1 erg = 1e-7 J, 1 J = 6.242e+12 MeV.
    Neutrons = np.empty(len(data['Bpeprd']))
    for timestep, energy in enumerate(data['Bpeprd']):
        # # Extended calculation that explicitly shows conversion between units.
        # EnergyInJules = sum(energy) * 1e-7
        # EnergyInMeV = EnergyInJules * 6.242e+12
        # EnergyInNeutronsMeV = 0.8 * EnergyInMeV
        # Neutrons[timestep] = EnergyInNeutronsMeV / 14.1

        # Single line replacement that replaces above calculations.
        Neutrons[timestep] = (0.8/(14.1 * 1.60218e-6))*sum(energy)

    if np.any(Neutrons > (max(Neutrons)*0.9999)):
        # Define the stagnation timestep as around the time when 99.99%  of neutrons have been produced.
        MaxStagnationTimestep = np.where(Neutrons > (max(Neutrons)*0.9999))[0][0]

        # Laser energy that has enetered the systems integrated up to the current timestep. erg -> Joules (1e-7). When accounting for CBET energy loss, real experiment will require E/CBET_MULTIPLIER to acheive HYADES result.
        LaserEnergy  = np.array(data['Elasin'])*(1e-7)/CBET_MULTIPLIER
        RequiredLaserEnergy = LaserEnergy[MaxStagnationTimestep]

        AH_energy = AH_energy_J / AH_efficiency

        TotalInjectedEnergy = RequiredLaserEnergy + AH_energy

        NeutronEnergy = 14.06e6*1.602E-19*max(Neutrons)
    
        Yeild = (sum(data['Bpeprd'][-1]) * 1e-7)

        NeutronGain = NeutronEnergy / TotalInjectedEnergy
        TotalGain = Yeild / TotalInjectedEnergy

    else:
        TotalGain = 0.0
        NeutronGain = 0.0

    return TotalGain, NeutronGain, Yeild


def ConvergenceRatio(file):
    """ 
    Calculates the convergence ratio of the implosion at each timestep. Olson convergence ratio is defined as the inner shell radius divided by the initial radius of the inner edge of the CH ablator. Lindl convergence ratio is defined as the inner shell radius divided by the initial radius of the whole capsule (outer edge of the CH ablator). Take the max of these time series to determine the CR of the whole implosion. 
    
    Parameters:
    file (string): path to file containing Hyades data.
    
    Returns:
    OlsonCRTimeSeries (1d array, float): Olson gain at each timestep.
    LindlCRTimeSeries (1d array, float): Lindl gain at each timestep.
    """
    data = nc.Dataset(file)

    Density = np.array(data['Rho'])
    X = np.array(data['R'][:-1])

    InnerShellRadii = ShellParameters(file)[1]
    OuterShellRadii = ShellParameters(file)[3]

    # Ice (foam) boundary
    InitialIceBoundaryIndex = np.where(Density[0] > 0.211)[0][0]
    InitialIceBoundaryRadius = X[0][InitialIceBoundaryIndex]

    # CR at all timesteps
    OlsonCRTimeSeries = InitialIceBoundaryRadius / InnerShellRadii
    LindlCRTimeSeries = X[0][-1] / InnerShellRadii

    # Max CR.
    OlsonCR = max(OlsonCRTimeSeries)
    LindlCR = max(LindlCRTimeSeries)

    return OlsonCRTimeSeries, LindlCRTimeSeries



def IFAR(file):
    """ 
    Calculates the in-flight aspect ratio of the implosion at each timestep. The in-flight aspect ratio is defined as the outer shell radius divided by the shell thickness 

    shell outer radius / (shell outer radius - shell inner radius) 
    
    Also returns the in-flight aspect ratio when the shell outer radius is at two-thirds of the initial radius (usually the stated IFAR value of an implosion). 
    
    Parameters:
    file (string): path to file containing Hyades data.
    
    Returns:
    IFARTimeSeries(1d array, float): IFAR at each timestep.
    IFAR (float): IFAR at measurement timestep
    """

    # Defined as ablation front radius (outer shell radius) divided by shell thickness, at the time where ablation front radius is at 2/3 of the initial inner radius of the shell (measurement_time).
    data = nc.Dataset(file)

    # Initial density and radii values (timestep 0). Rcm is used with ignoring first and last index.
    InitialDensity = data['Rho'][0]
    InitialRadius = data['R'][0][:-1]

    # Initial inner shell radius
    truth_array = InitialDensity > 0.211
    InnerIceIndex = np.where(truth_array)[0][0]
    InnerIceRadius = InitialRadius[InnerIceIndex]

    # Time series of the shell inner and outer radius.
    InnerShellRadiusTimeseries = ShellParameters(file)[1]
    OuterShellRadiusTimeseries = ShellParameters(file)[3]

    # IFAR time series
    IFARTimeSeries = OuterShellRadiusTimeseries / (OuterShellRadiusTimeseries - InnerShellRadiusTimeseries)

    # Calculate the measurement time, when the ablation front is 2/3 of the initial outer ice radius.
    truth_array = OuterShellRadiusTimeseries < ( (2 / 3) * InnerIceRadius)
    measurement_time_index = np.where(truth_array)[0][0]

    # IFAR at measurement time. 
    IFAR = IFARTimeSeries[measurement_time_index]

    return IFARTimeSeries, measurement_time_index


def ImplosionVelocity(file):
    """ 
    Calculates the implosion velocity at each timestep. The implosion velocity can be defined as the mass averaged velocity of the shell region or the velocity of the outer shell boundary. 
    
    Parameters:
    file (string): path to file containing Hyades data.
    
    Returns:
    ShellAveragedVelocityTimeSeries (1d array, float): Implosion velocity calculated using the mass averaged shell velcoity.
    ShellOuterVelocityTimeSeries (1d array, float): Implosion velocity calculated using the velocity of the outer shell boundary.
    """

    data = nc.Dataset(file)

    VolumeTimeSeries = np.array(data['Vol'])
    DensityTimeSeries = np.array(data['Rho'])
    MassTimeSeries = DensityTimeSeries * VolumeTimeSeries[:, 1:]    # g

    VelocityTimeSeries = np.array(data['U']) / 100000               # km/s

    InnerShellIndexTimeSeries = ShellParameters(file)[0]
    OuterShellIndexTimeSeries = ShellParameters(file)[2]

    ShellAveragedVelocityTimeSeries = []
    ShellOuterVelocityTimeSeries = []
    for timestep in range(len(VolumeTimeSeries)):
        InnerShellIndex = int(InnerShellIndexTimeSeries[timestep])
        OuterShellIndex = int(OuterShellIndexTimeSeries[timestep])
        
        Mass = MassTimeSeries[timestep][InnerShellIndex:OuterShellIndex]
        Volume = VolumeTimeSeries[timestep][InnerShellIndex:OuterShellIndex]

        Velocity = VelocityTimeSeries[timestep][InnerShellIndex+1:OuterShellIndex+1]

        # Mass averaged velocity
        ShellAveragedVelocity = sum(Mass * Velocity) / sum(Mass)
        # Volume averaged velocity
        # ShellAveragedVelocity = sum(Volume * Velocity) / sum(Volume)

        ShellAveragedVelocityTimeSeries.append(ShellAveragedVelocity)

        ShellOuterVelocityTimeSeries.append(VelocityTimeSeries[timestep][OuterShellIndex+1])

    ImplosionVelocity = min(ShellAveragedVelocityTimeSeries)

    return ShellAveragedVelocityTimeSeries, ShellOuterVelocityTimeSeries


def RhoR(file):
    """ 
    Calculates the rhoR of the hotspot, shell, whole capsule and the compressed region (hotspot + shell). The rhoR is the density integrated radially.
    
    Parameters:
    file (string): path to file containing Hyades data.
    
    Returns:
    CapsuleRhoR (1d array, float): RhoR of the whole capsule.
    ShellRhoR (1d array, float): RhoR of the shell.
    HotspotRhoR (1d array, float): RhoR of the hotspot.
    CompressedRhoR (1d array, float): RhoR of the hotspot and shell (ignores coronal plasma).
    """

    data = nc.Dataset(file)

    Density = np.array(data['Rho'])
    Radius = np.array(data['Rcm'])[:, 1:-1]

    ShellInnerIndex = ShellParameters(file)[0]
    ShellOuterIndex = ShellParameters(file)[2]

    CapsuleRhoR = np.trapz(Density, Radius)

    ShellRhoR = np.empty(len(CapsuleRhoR))
    HotspotRhoR = np.empty(len(CapsuleRhoR))
    CompressedRhoR = np.empty(len(CapsuleRhoR))

    for timestep in range(len(Density)):
        inner_index = int(ShellInnerIndex[timestep])
        outer_index = int(ShellOuterIndex[timestep])

        density = Density[timestep]
        radius = Radius[timestep]

        shell_rho_r_timestep = np.trapz(density[inner_index:outer_index], radius[inner_index:outer_index])
        
        hotspot_rho_r_timestep = np.trapz(density[:inner_index], radius[:inner_index])

        compressed_rho_r_timestep = np.trapz(density[:outer_index], radius[:outer_index])

        ShellRhoR[timestep] = shell_rho_r_timestep
        HotspotRhoR[timestep] = hotspot_rho_r_timestep
        CompressedRhoR[timestep] = compressed_rho_r_timestep 

    return CapsuleRhoR, ShellRhoR, HotspotRhoR, CompressedRhoR


def LaserProfile(file, CBET_MULTIPLIER=0.8):
    """ 
    Produces the laser power profile used in the simulation. 

    Parameters:
    file (string): path to file containing Hyades data.
    CBET_MULTIPLIER (0.0 < float < 1.0): fraction of laser energy remaining after CBET losses.

    Returns:
    SimulationLaserPower (1d, float): laser power profile used in the simulation
    ExperimentalLaserPower (1d, float): laser power profile required for equivalent experimental implosion with CBET.
    ParametricLimit (1d, float): maximum laser intensity
    """

    data = nc.Dataset(file)
    Time = data['DumpTimes']
    InitialRadius = data['R'][0][-1]

    # Find the number of neutrons released throughout the simulation (integrated up to the current timestep). 1 erg = 1e-7 J, 1 J = 6.242e+12 MeV.
    Neutrons = np.empty(len(data['Bpeprd']))
    for timestep, energy in enumerate(data['Bpeprd']):
        # # Extended calculation that explicitly shows conversion between units.
        # EnergyInJules = sum(energy) * 1e-7
        # EnergyInMeV = EnergyInJules * 6.242e+12
        # EnergyInNeutronsMeV = 0.8 * EnergyInMeV
        # Neutrons[timestep] = EnergyInNeutronsMeV / 14.1

        # Single line replacement that replaces above calculations.
        Neutrons[timestep] = (0.8/(14.1 * 1.60218e-6))*sum(energy)

    if np.any(Neutrons > (max(Neutrons)*0.9999)):
        # Define the stagnation timestep as around the time when 99.99%  of neutrons have been produced.
        MaxStagnationTimestep = np.where(Neutrons > (max(Neutrons)*0.9999))[0][0]

        # Laser energy that has enetered the systems integrated up to the current timestep. erg -> Joules (1e-7). When accounting for CBET energy loss, real experiment will require E/CBET_MULTIPLIER to acheive HYADES result.
        SimulationLaserEnergyJoules  = np.array(data['Elasin'])*(1e-7)
        SimulationRequiredLaserEnergy = SimulationLaserEnergyJoules[MaxStagnationTimestep]

    SimulationLaserPower = np.array([0] + list((np.diff(SimulationLaserEnergyJoules) / np.diff(Time))))

    ExperimentalLaserPower = SimulationLaserPower / CBET_MULTIPLIER
    ParametricLimit = max(ExperimentalLaserPower) * 0.35**2 / ( 4*con.pi * InitialRadius**2)

    return SimulationLaserPower, ExperimentalLaserPower, ParametricLimit, SimulationRequiredLaserEnergy


def LaserEfficiency(file):
    """ 
    Calculates the amount of laser energy absorbed by the target.

    Parameters:
    file (string): path to file containing Hyades data.
    
    Returns:
    total_laser_energy (float): total laser energy injected into the simulation
    total_energy_deposition (float): total laser energy absorbed by the plasma
    """

    data = nc.Dataset(file)

    timesteps = np.array(data['DumpTimes'])
    total_laser_energy = np.array(data['Elasin'])[-1] * 1e-7

    deposited_laser_power = np.array(data['Deplas']) * 1e-7
    total_deposited_power_per_timestep = np.sum(deposited_laser_power, axis=1)
    total_energy_deposition = np.trapz(total_deposited_power_per_timestep, timesteps)

    return total_laser_energy, total_energy_deposition



def EnergyEvolution(file):
    """
    Compares the thermal, kinetic and total energy of the simulation (in kJ). 

    Parameters:
    file (string): path to file containing Hyades data.

    Returns:
    thermal_energy (1d, float): thermal energy at each timestep (in kJ)
    kinetic_energy (1d, float): kinertic energy at each timestep (in kJ)
    total_energy (1d, float): total energy at each timestep (in kJ)
    """
    data = nc.Dataset(file)

    thermal_energy = np.array(data['Eelct']) / 1e10
    kinetic_energy = np.array(data['Ekint']) / 1e10
    total_energy = thermal_energy + kinetic_energy

    return thermal_energy, kinetic_energy, total_energy



def DensityTemperatureContour(file, STATIC_RESOLUTION=10000):
    """
    The simulation uses a Lagrangian grid, so need to interpolate this onto a static grid to create contour plots. This function creates the static grid and interpolates density/temperature values onto it.

    Parameters:
    - file (string): path to the netCDF file.
    - STATIC_RESOLUTION (int): number of grid points of the static mesh to interpolate values onto. 

    Returns:
    - radius_interpolate (1D, float): static radius grid that values are interpolated at.
    - static_density (2D, float): density interpolated onto a static grid at each timestep
    - temperature_interpolate (2D, float): temperature interpolated over the radius mesh at each timestep. 
    """
    data = nc.Dataset(file)

    # Initial cell centre radii 
    initial_centres = np.array(data['Rcm'])[0][1:-1]

    # Create static grid with high resolution.
    static_radius = np.linspace(0, initial_centres[-1], STATIC_RESOLUTION, endpoint=True)

    # Create an empty 2d array to interpolate temperature/density values into.
    static_temperature = np.zeros( (len(np.array(data['DumpTimes'])), STATIC_RESOLUTION) )
    static_density = np.zeros( (len(np.array(data['DumpTimes'])), STATIC_RESOLUTION) )

    for i in range(len(np.array(data['DumpTimes']))):
        static_temperature[i, :] = np.interp(static_radius, np.array(data['Rcm'])[i][1:-1], np.array(data['Te'])[i])
        static_density[i, :] = np.interp(static_radius, np.array(data['Rcm'])[i][1:-1], np.array(data['Rho'])[i])

    return static_radius, static_density, static_temperature



def Adiabat(file):
    """
    !!! Write this !!! 

    Parameters:
    - file (string): path to the netCDF file.
    - STATIC_RESOLUTION (int): number of grid points of the static mesh to interpolate values onto. 

    Returns:
    - adiabat (2D, float): adiabat over the whole domain at each timestep.
    """
    data = nc.Dataset(file)

    pressure = np.array(data['Pres']) / 1e12 # Pressure in Mbar

    density = np.array(data['Rho'])

    adiabat = pressure / (2.18 * density**(5/3))

    return adiabat
