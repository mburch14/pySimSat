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

    def __init__(self, detector, orbit, solmod = 0.5):
        self.detector = detector
        self.inclination = orbit.inclination
        self.altitude = orbit.altitude
        self.solmod = solmod

    def gen_spectrum_table(self, output, dcxr, albedo, cralbedo, cosmicrays):
        energies = self.detector.energy #Gives energy bin midpoints in keV
        energy_lo = self.detector.energy_low #Gives energy bin lower bounds in keV
        energy_hi = self.detector.energy_high #Gives energy bin upper bounds in keV

        #flux in units of photons/cm2/s/keV
        fluxes = 0
        if dcxr:
            fluxes += np.array(self.dcxr(energies, self.detector.geos.fov_sr))
        if albedo:
            fluxes += np.array(self.albedo(energies, self.detector.geos.fov_sr))
        if cralbedo:
            fluxes += np.array(self.cosmicrayalbedo(energies, self.detector.geos.fov_sr, self.solmod))
        if cosmicrays:
            fluxes += 0.33 * np.array(self.dcxr(energies, 0.67))

        table = np.column_stack((energy_lo, energy_hi, fluxes))
        np.savetxt(output, table, fmt="%.6f %.6f %.8e", comments="")
        return output

    def dcxr(self, energy, fov_sr):
        #This is from Insight-HXMT measurements of the diffuse X-ray background by Huang et al. 2022
        #In units of photons/cm2/s/keV
        intensity = 9.67 * energy**-1.33

        return intensity * fov_sr

    def albedo(self, energy, fov_sr):

        #Albedo spectrum from churazov et al. Earth X-ray albedo for cosmic X-ray background radiation in the 1–1000 keV band
        #In units of photons/cm2/s/keV
        energy = np.asarray(energy)
        term1 = 1.22/ (((energy/28.5)**-2.54) + ((energy/51.3)**1.57) - 0.37)
        term2 = (2.93 + (energy/3.08)**4) / (1 + (energy/3.08)**4)
        term3 = (0.123 + (energy/91.83)**3.44) / (1 + (energy/91.83)**3.44)
        cxbint = self.dcxr(energy, fov_sr)
        
        return term1*term2*term3*cxbint

    def cosmicrayalbedo(self, energy, fov_sr, solmod):
        #Albedo spectrum from cosmic rays from Hard X-ray emission of the Earth’s atmosphere: Monte Carlo simulations by Sazonov et al. 2021
        #In units of photons/cm2/s/keV
        energy = np.asarray(energy)
        C = self.getCRalbedoC(solmod=solmod, inclination =self.inclination, altitude= self.altitude)
        intensity = C / ((energy/44)**-5 + (energy/44)**1.4)
        return intensity * fov_sr        
        
    def getCRalbedoC(self, inclination, altitude, solmod):
        #normalization constant for cosmicrayalbedo (units: 1/s/cm2/sr)

        #the geomagnetic latitude is approximated as |i/2|
        theta_M = np.deg2rad(inclination/2)
        earth_radius = 6371 #km
        #approximation of the geomagnetic cutoff in units GV
        R_cut = (14.5* (1+altitude/earth_radius)**-2) * (np.cos(theta_M))**4

        #equation in the Sazonov et al paper
        term1 = 1.47 * 0.0178
        term2 = ((solmod / 2.8)**0.4 + (solmod / 2.8)**1.5)**-1
        denom = (1.3 * (solmod**0.25) * (1 + 2.5 * solmod**0.4))
        term3 = (1 + (R_cut / denom)**2)**-0.5
        C = term1 * term2 * term3

        return C


class detector():
    def __init__(self, geometry, orbit, mission, optics, res, grad, low_ecut, material, mat_density, activedetector):
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
        self.activedetector = activedetector

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

        if self.optics.localized:
            Tmean = f * 1 + (1 - f) * tmask
            variance = f*(1 -Tmean)**2 + (1 -f)*(tmask- Tmean)**2
            coding_eff = np.sqrt(variance)

        
        #soft energy cutoff
        fwhm_at_ecut = ((self.low_ecut-1)*self.grad + self.res)/1000
        sigma = fwhm_at_ecut / (2*np.sqrt(2*np.log(2)))
        weight = 0.5 * (1 + erf((energy-self.low_ecut) / (np.sqrt(2)*sigma)))

        return coding_eff * weight * acoll * tdet * self.activedetector

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
    def __init__(self, thickness, mask_material, mask_density, localized):
        self.thickness = thickness
        self.localized = localized
        self.mask_element = mask_material
        self.mask_density = mask_density

    def transmission(self, energy):
        atten_const = xraydb.material_mu(self.mask_element, energy * 1000, density=self.mask_density) #Energy in kev. xraydb.mu_elam takes energy in eV, so we multiply by 1000 to convert from keV to eV.
        return np.exp(-atten_const * self.thickness * 0.1) #The 0.1 is to convert from mm to cm, since the thickness is in mm and the attenuation constant is in cm^-1
