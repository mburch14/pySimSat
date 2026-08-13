import math
import numpy as np
from astropy.io import fits
import xraydb
import subprocess
from scipy.special import erf
from pathlib import Path

class Mission:

    def __init__(self, name, energymin, energymax):
        self.name = name

        #Energy range in keV
        self.energymin = energymin
        self.energymax = energymax


class Orbit:

    def __init__(self, altitude, inclination):
        self.earthradius = 6378.137  # km
        self.altitude = altitude #km
        self.inclination = inclination #degrees

        self.theta = np.arcsin(self.earthradius / (self.earthradius + altitude)) #radians, angle between the zenith and the horizon.

        #The fraction of the sky blocked by the Earth, assuming a circular orbit and a spherical Earth.
        self.earth_blocking_fraction = 0.5 * (1 - np.cos(self.theta))


class geometry:

    def __init__(self, config):

        #dimension of the detector in cm
        self.detl = config["detl"]
        self.detw = config["detw"]
        self.detthickness = config["dett"] #thickness of the detector
        self.detsep = config["detsep"] #seporation between mask and detector

        #represented as a decimal between 0 and 1, where 1 is 100% transmission and 0 is 0% transmission. This is the fraction of open slots in mask (Does not consider optics).
        self.maskOpen = config["maskOpen"]

        #dimension of the cubesat in cm
        self.maskl = config["maskl"]
        self.maskw = config["maskw"]
        self.maskh = config["maskh"]
        self.collimator = config["collimator"] #True if the cubesat is using a collimator, False if it is using a mask. If it is using a collimator, the field of view will be calculated using the formula for the solid angle of a rectangle with the dimensions of the mask, rather than the dimensions of the detector.

        if self.collimator:

            self.coll = config["coll"] #cm, collimator length
            self.colw = config["colw"] #cm, collimator width

            #fully coded field of view in the width direction; fully coded field of view in the length direction
            self.wfov = 2 * math.atan(self.colw / self.detsep) #in radians, inside tan may be divided by 2
            self.lfov = 2 * math.atan(self.coll / self.detsep) #in radians

            #half coded field of view in the width direction; half coded field of view in the length direction
            #self.half_coded_w = 2 * math.atan(self.colw / self.detsep) #in radians
            #self.half_coded_l = 2 * math.atan(self.coll / self.detsep) #in radians

        else:
            #wfov is the field of view in the width direction; lfov is the field of view in the length direction
            self.wfov = 2*math.atan((self.maskw -self.detw) / (2*(self.detsep))) #in radians
            self.lfov = 2*math.atan((self.maskl -self.detl) / (2*(self.detsep))) #in radians

            #half coded field of view in the width direction; half coded field of view in the length direction
            self.half_coded_w = 2 * math.atan(self.maskw / (2*(self.detsep))) #in radians
            self.half_coded_l = 2 * math.atan(self.maskl / (2*(self.detsep))) #in radians

        #total field of view in steradians, calculated using the formula for the solid angle of a rectangle
        self.fov_sr = 4 * math.asin(np.sin(self.wfov / 2) * np.sin(self.lfov / 2))

        try:
            self.half_coded_fov = 4 * math.asin(np.sin(self.half_coded_w / 2) * np.sin(self.half_coded_l / 2))
        except:
            pass

        #collecting area in cm^2; assuming that the photons are othogonal to the detector. if not safe assumption, multiply by cos(theta).
        self.collecting_area = self.maskOpen * self.detl * self.detw


class BackgroundModel:

    def __init__(self, detector):
        self.detector = detector


    def F_M(self, energy, Mc2, Z, phi):
        #energy: particle kinetic energy, GeV
        #Mc2: particle rest mass energy, GeV
        #Z: charge, unitless
        #phi: solar modulation factor, GV, 0.55GV at solar minimum, 1.1 for GV at solar maximum
        return ((energy + Mc2)**2 - Mc2**2)/((energy + abs(Z)*phi + Mc2)**2 - Mc2**2)
    
    def geomagnetic_cutoff(self, R, altitude, theta_M, r):
        R_E = 6371 #km
        R_cut = (14.5* (1+altitude/R_E)**-2) * (np.cos(theta_M))**-2 #GV
        return 1/ (1 + (R/R_cut)**-r)
    
    def R_E (self, E, mc2, Z):
        energy = np.asarray(E)
        pc = np.sqrt((energy + mc2)**2 - mc2**2) #GeV
        return pc/abs(Z) #GV4

    def gen_spectrum_table(self, output, cxb, albedo, particle):
        energies = self.detector.energy #Gives energy bin midpoints in keV
        energy_lo = self.detector.energy_low #Gives energy bin lower bounds in keV
        energy_hi = self.detector.energy_high #Gives energy bin upper bounds in keV

        #flux in units of photons/cm2/s/keV
        fluxes = 0
        if cxb:
            fluxes = 1.33 * np.array(self.cxb(energies, self.detector.geos.fov_sr))
        if albedo:
            fluxes += self.albedo(energies, self.detector.geos.fov_sr)
        if particle:
            fluxes += 0 #not done yet

        table = np.column_stack((energy_lo, energy_hi, fluxes))
        np.savetxt(output, table, fmt="%.6f %.6f %.8e", comments="")
        return output

    def cxb(self, energy, fov_sr):
        #Cosmic X-ray background spectrum from Gruber et al. 1999, ApJ, 520, 124
        #In units of photons/cm2/s/keV
        C = 10.15e-2
        EB = 29.99  # keV
        gamma1 = 1.32
        gamma2 = 2.88
        energy = np.asarray(energy)
        intensity = C / ((energy / EB) ** gamma1 + (energy / EB) ** gamma2)
        return intensity * fov_sr

    def albedo(self, energy, fov_sr):
        #Albedo spectrum from Ajello et al. 2008, ApJ, 689, 666
        #In units of photons/cm2/s/keV
        EB = 33.7  #in keV
        Gamma1 = -5
        Gamma2 = 1.72
        const = 1.48e-2
        energy = np.asarray(energy)
        intensity = const / ((energy / EB) ** Gamma1 + (energy / EB) ** Gamma2)
        return intensity * fov_sr


class ChargedParticles(BackgroundModel):
    def __init__(self, detector, inclination, altitude, phi):
        super().__init__(detector)
        self.inclination = inclination
        self.altitude = altitude
        self.phi = phi

    def primaryIntensity(self, fov_sr):
        return self.protonIntensity(fov_sr) + self.electronpositronIntensity(fov_sr) + self.alphaIntensity(fov_sr)
    
    def protonIntensity(self, fov_sr):
        #Taken from Background simulations for the Large Area Detector onboard LOFT (Campana et al, 2013)
        E = np.logspace(-3, 2, 1000)#GeV
        E_IS = E + self.phi #This is for F(E + Z\psi)
        M_pc2 = 0.938 #GeV, rest mass energy of a proton
        R_E = self.R_E(E, M_pc2, 1) #GV, the last is Z, which is 1 for proton
        R_IS = self.R_E(E_IS, M_pc2, 1)  # for FU
        C = self.geomagnetic_cutoff(R_E, self.altitude, np.deg2rad(self.inclination/2), r = 12) #unitless; r value is 12 for protons; theta_M is approximated as i/2???
        F_M = self.F_M(energy=E, Mc2=M_pc2, Z=1, phi=self.phi) #unitless; need to fix phi!!
        F_U = 1e-7 * 23.9 * R_IS**-2.83 # particles/cm2/s/sr/keV
        return F_U * F_M * C * fov_sr
    
    def electronpositronIntensity(self, fov_sr):
        #Taken from Background simulations for the Large Area Detector onboard LOFT (Campana et al, 2013)
        E = np.logspace(-3, 2, 1000)#GeV
        E_IS = E + self.phi #This is for F(E + Z\psi)
        M_pc2 = 0.00051 #GeV, electron rest mass energy
        R_E = self.R_E(E, M_pc2, 1) #GV, the last is Z, which has a magnitude of 1 for electrons and positrons I DO NOT KNOW IF IT IS SUPPOSED TO BE THE ABSOLUTE VALUE OR NOT!!!
        R_IS = self.R_E(E_IS, M_pc2, 1)  # for FU
        C = self.geomagnetic_cutoff(R_E, self.altitude, np.deg2rad(self.inclination/2), r = 6) #unitless; r value is 6 for electrons and positrons; theta_M is approximated as i/2???
        F_M = self.F_M(energy=E, Mc2=M_pc2, Z=1, phi=self.phi) #unitless; need to fix phi!!!
        F_Uneg = 1e-7 * 0.65 * R_IS**-3.3 # particles/cm2/s/sr/keV
        F_Upos = 1e-7 * 0.051 * R_IS**-3.3 # particles/cm2/s/sr/keV
        return (F_Uneg + F_Upos) * F_M * C * fov_sr
    
    def alphaIntensity(self, fov_sr):
        #Taken from Background simulations for the Large Area Detector onboard LOFT (Campana et al, 2013)
        E = np.logspace(-3, 2, 1000)#GeV
        E_IS = E + 2*self.phi #This is for F(E + Z\psi)
        M_pc2 = 3.727 #GeV, alpha particle (2proton, 2 neuton) rest mass energy
        R_E = self.R_E(E, M_pc2, 2) #GV, the last is Z, which has a magnitude of 2 for ionized helium
        R_IS = self.R_E(E_IS, M_pc2, 2)  # for FU
        C = self.geomagnetic_cutoff(R_E, self.altitude, np.deg2rad(self.inclination/2), r = 12) #unitless; r value is 12 for helium; theta_M is approximated as i/2???
        F_M = self.F_M(energy=E, Mc2=M_pc2, Z=2, phi=self.phi) #unitless; need to fix phi!!!
        F_U = 1e-7 * 1.5 * R_IS**-2.7 # particles/cm2/s/sr/keV 
        return F_U * F_M * C * fov_sr


class detector():
    def __init__(self, geometry, orbit, mission, optics, res, grad, low_ecut, material, mat_density):
        self.geos = geometry
        self.orbs = orbit
        self.missions = mission
        self.optics = optics
        self.energy_edges = np.linspace(self.missions.energymin, self.missions.energymax, self.missions.energymax-self.missions.energymin+1)   #1 keV bins
        self.energy_low = self.energy_edges[:-1] #zeroth element to second-to-last element
        self.energy_high = self.energy_edges[1:] #first element to last element
        self.energy = 0.5*(self.energy_low + self.energy_high) #Midpoint of each energy bin.
        self.res = res
        self.grad = grad
        self.low_ecut = low_ecut
        self.material_formula = material
        self.material_density = mat_density

    def effective_area(self, energy):
        
        #Add in the likelihood that some photons may be transmitted through the mask.
        acoll = self.geos.collecting_area

        coding_eff = 1

        #Energy in kev. xraydb.mu_elam takes energy in eV, so we multiply by 1000 to convert from keV to eV.
        atten_const = xraydb.material_mu(self.material_formula, energy * 1000, density=self.material_density)
        tdet = 1-np.exp(-atten_const * self.geos.detthickness * 0.1) #The 0.1 is to convert from mm to cm, since the thickness is in mm and the attenuation constant is in cm^-1.

        f = self.geos.maskOpen
        tmask = self.optics.transmission(energy)
        acoll += tmask *(1- f)*self.geos.detl*self.geos.detw

        if self.optics.focusing:
            Tmean = f * 1 + (1 - f) * tmask
            variance = f*(1 -Tmean)**2 + (1 -f)*(tmask- Tmean)**2
            coding_eff = np.sqrt(variance)

        
        #soft energy cutoff
        fwhm_at_ecut = ((self.low_ecut-1)*self.grad + self.res)/1000
        sigma = fwhm_at_ecut / (2*np.sqrt(2*np.log(2)))
        weight = 0.5 * (1 + erf((energy-self.low_ecut) / (np.sqrt(2)*sigma)))

        return coding_eff * weight * acoll * tdet

    def det_absorption(self, energy):
        #Energy in kev. xraydb.mu_elam takes energy in eV, so we multiply by 1000 to convert from keV to eV.
        atten_const = xraydb.material_mu(self.material_formula, energy * 1000, density=self.material_density)
        return 1-np.exp(-atten_const * self.geos.detthickness * 0.1) #The 0.1 is to convert from mm to cm, since the thickness is in mm and the attenuation constant is in cm^-1.

    def gen_arf(self,arf, energy_lo=None, energy_hi=None):
        if energy_lo is None or energy_hi is None:
            energy_edges = np.linspace(self.missions.energymin, self.missions.energymax, self.missions.energymax - self.missions.energymin + 1)
            energy_lo = energy_edges[:-1]
            energy_hi = energy_edges[1:]

        energy = 0.5 * (energy_lo + energy_hi)
        aeff = np.array([self.effective_area(e) for e in energy])
        
        #This is the actual ARF Matrix. Col1 is the lower energy bound, Col2 is the upper energy bound, and Col3 is the effective area in cm^2.
        cols = [
            fits.Column(name='ENERG_LO', format='D', unit='keV', array=energy_lo),
            fits.Column(name='ENERG_HI', format='D', unit='keV', array=energy_hi),
            fits.Column(name='SPECRESP', format='D', unit='cm**2', array=aeff)]
        
        arf_hdu = fits.BinTableHDU.from_columns(cols)
        arf_hdu.name = "SPECRESP"

        #These are the required header keywords for an ARF file, according to the OGIP standard. (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc4)
        hdr = arf_hdu.header
        hdr["HDUCLASS"] = "OGIP"
        hdr["HDUCLAS1"] = "RESPONSE"
        hdr["HDUCLAS2"] = "SPECRESP"
        hdr["HDUVERS"]  = "1.1.0"
        hdr["CHANTYPE"] = "PI"
        hdr["HDUCLAS3"] = "FULL"
        hdr["TELESCOP"] = f"{self.missions.name}"
        hdr["INSTRUME"] = "CZT"
        hdr["FILTER"]   = "NONE"
        hdr["EXTNAME"]  = "SPECRESP"

        primary = fits.PrimaryHDU()
        hdul = fits.HDUList([primary, arf_hdu])
        hdul.writeto(arf, overwrite=True) #The name of the output ARF file is stored in arf. This will be used to generate the RMF file.
        return arf

    def gen_rsp(self, arf, rsp):
        resolution = self.res #keV FWHM at 1 keV, in eV
        gradient = self.grad #eV/keV, the change in resolution with energy
        subprocess.run(["ogipgenrsp", #This is the xspec command to generate an RMF file from an ARF file. It is part of the HEASoft package.
            "--arffile", arf, #The arf should be generated first, and then the rmf can be generated from it.
            "--resolution", str(resolution), 
            "--resgradient", str(gradient),
            "--rspfile", rsp, #The name of the output RMF file
            "--range", f"{self.missions.energymin}:{self.missions.energymax}",
            "--overwrite"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,) #stdout is the normal output, stderr is the error output. We are redirecting both to DEVNULL to suppress the output of the command.


class optics():
    def __init__(self, thickness, mask_material, mask_density, focusing):
        self.thickness = thickness
        self.focusing = focusing
        self.mask_element = mask_material
        self.mask_density = mask_density

    def transmission(self, energy):
        atten_const = xraydb.material_mu(self.mask_element, energy * 1000, density=self.mask_density) #Energy in kev. xraydb.mu_elam takes energy in eV, so we multiply by 1000 to convert from keV to eV.
        return np.exp(-atten_const * self.thickness * 0.1) #The 0.1 is to convert from mm to cm, since the thickness is in mm and the attenuation constant is in cm^-1
