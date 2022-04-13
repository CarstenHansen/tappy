#!/usr/bin/env python2
# -*- coding: utf-8 -*-

""" Carsten Hansen, fcoo.dk

    Analyze tidal harmonics for a list of locations from hindcasts with an
    estuarine 3d hydrodynamic model (GETM) over 5+ years, and predict the
    tides elevation timeseries for some years ahead
"""

""" Based initially on pwcazenave/tappy readme.md
"""
import numpy as np
from netCDF4 import Dataset,num2date,date2num,stringtochar
from tappy import tappy
from  datetime import datetime
from math import floor
import os, sys, copy
import cPickle as pickle
import time

""" Setup """

# Input base directory
filebase = '[See earlier versions]' + '/'
# Output base directory 
outbase = '[See earlier versions]' + '/'

years_analyze = [ 2017, 2018, 2019, 2020, 2021 ]
years_predict = [ 2022, 2023, 2024, 2025, 2026 ]

modelid = 'NS1C-v014Eey'
setup_run = 'NS1C-v014Eey-sf2103avc60036x0030'
delta_time = 10 # minutes

# Data input file template and method to substitute each year and station
filetempl = setup_run + '/cat_[YEAR]/ST_[STATION]_2Dvars_' \
             + '%dmin.' % delta_time + setup_run + '-rcat_[YEAR].nc'

def filein_set( year, STATION):
    filein = filetempl[:]
    for r in ('[YEAR]', '%d' % year), ('[STATION]', STATION):
        filein = filein.replace(*r)
    return filebase, filein

# Analyzed constituents dictionary to be saved in a pickle file harmonics.pkl
harmonics_pkl = 'harmonics.pkl'

# Identifier for each harmonics analysis
setup_anal = modelid + '_%d-%d' % (years_analyze[0],years_analyze[-1])

if years_predict:
    # Predictions output directory construction
    setup_pred = '%dmin.' % delta_time + modelid \
                 + '_' + '-'.join([str(yr) for yr in years_predict])
    # Predictions output file (each STATION name to be substituted)
    outfile_templ = 'Tides_[STATION]_'+setup_pred+'.nc'
    
modelid = 'GETM %s'%modelid

outdir = outbase + 'GETM_'+setup_anal+'/'

# Astronomics package output directory
astrodir = outdir

"""
To get STATIONS list from model output directory
-- See at end -- 
"""

STATIONS=["A121TG","AABENRAA","AARHUS",] # etc.

delta_time *= 60. # 10 min

num_years = 0 # If len(years_predict)>0: Use existing analysis results from
# file to make prediction.
# If years_predict==[]: Make analyses shorter that 1 year
# If there are no analysis results, we will set 
# num_years = len(years_analyze) in the loop below to do the analysis
# and then do the prediction with num_years = 0

# In the logics of the main routine we presume that:
if num_years > 0:
    # Then don't calculate a prediction
    assert years_predict == []

# Read in previously analyzed constituents, to e.g. extend the predictions
try:
    with open(harmonics_pkl,'r') as stfil:
        r_ph = pickle.load(stfil)
except:
    r_ph = dict()

if not setup_anal in r_ph.keys():
    r_ph[setup_anal] = dict()


"""
About the OSU constituents used in the NS1C model boundary conditions for
elevation and transports/currents:

"The TPXO models include complex amplitudes of MSL-relative sea-surface
elevations and transports/currents for eight primary (M2, S2, N2, K2, K1,
O1, P1, Q1), two long period (Mf,Mm) and 3 non-linear (M4, MS4, MN4)
harmonic constituents (plus 2N2 and S1 for TPXO9 only).

TPXO9 versions starting from v2 and v5a include minor constituents:
2Q1, J1, L2, M3, MU2, NU2, OO1."

"""

# Restrict to the possible OSU constituents + possible inferred constituents
HARMall = ["M2", "O1", "S2", "Mf", "Mm", "Q1", "K2",  "K1", "N2", "P1",
           "S1", "2Q1",  "J1",  "L2", "mu2", "nu2", "OO1",
           "2N2", "MN4","M4", "M6", "M8", "MS4", "2MS6" ]

# For short analysis ( <~ 28 days )
HARMshort = ["M2", "K1", ]

# In longer analysis
HARMlong = [ "O1", "S2", "N2", "Q1", "K2", "Mf", "Mm",   ]

# In very long analysis
HARMminor = [ "P1", "S1", "2Q1",  "J1",  "L2", "mu2", "nu2", "OO1" ]

HARMnonlin = [ "M4","2N2", "MS4", "MN4", "M6", "M8", "2MS6" ]
# Ignore M3; add M6, M8, "2MS6" as inferred

HARManalyse = HARMshort
HARMlong += HARMminor + HARMnonlin
assert len(set(HARMall)) == len(set(HARManalyse)) + len(set(HARMlong))
assert set(HARMall) == set(HARManalyse + HARMlong)
assert len(HARMall) == len(HARManalyse) + len(HARMlong)

# Declare a global parameters
regs = {'seriesfile': '',      # Name of timeseries netcdf file
        'times_def': [[], '',0 ], # time, time_units, delta_time
       }
regs['times_def'][2] = delta_time

if years_predict:
    # Construct the prediction times and dates
    pred_start = datetime(years_predict[0]-1, 12, 16, 0)
    pred_end = datetime(years_predict[-1]+1, 1, 16, 0)
    pred_time_units = 'seconds since %04d-01-01 00:00:00' % years_predict[0]            
    time0 = date2num(pred_start,pred_time_units)
    time1 = date2num(pred_end,pred_time_units)
    series_len = int(round((time1 - time0)/delta_time)) + 1

    pred_times = np.arange(0,series_len) * delta_time + time0
    pred_dates = num2date(pred_times,pred_time_units)

    pred_years_id = '%d-%d' %(years_predict[0],years_predict[-1])
        
def pred_to_netCDF(STATION, ncfile, x, harm, rr, ph, pred_times, pred_all):
    # Prediction timeseries to netCDF file
    
    f = Dataset(ncfile,"w", format="NETCDF3_CLASSIC")
    f.createDimension("time",None)
    f.createDimension('constit', len(harm))
    f.createDimension('strlen',4)

    times = f.createVariable("time","f8",("time",))
    times[:] = pred_times
    times.long_name = 'time'
    times.units = pred_time_units
    times.calendar ='standard'
    
    myvar = f.createVariable('tides','f4',('time',))
    myvar[:] = pred_all[:]
    myvar.units = 'm'
    myvar.standard_name = 'tidal_sea_surface_height_above_mean_sea_level'
    myvar.long_name = 'sea surface tide '+STATION
    myvar.setncattr('station',STATION)

    cname = f.createVariable('constituent','S1',('constit','strlen'))
    amplit = f.createVariable('constit_amplitude','f4',('constit',))
    phase = f.createVariable('constit_phase','f4',('constit',))
    amplit.units = 'm'
    phase.units = 'degrees'
    
    cname[:] = stringtochar(np.array(harm,'S'))
    amplit[:] = np.asarray( [rr[constit] for constit in harm] )
    phase[:] = np.asarray( [ph[constit] for constit in harm] )
    
    f.history = "Created for FCOO at " + time.ctime(time.time())
    f.setncattr('Model',modelid)
    f.setncattr('Tide_analysis','https://github.com/pwcazenave/tappy accessed 2022-04-05')
    f.setncattr('Analysis_years','%d-%d'%(years_analyze[0],years_analyze[-1]))
    f.setncattr('Tappy-kw_linear_trend','%s'%x.linear_trend)
    f.setncattr('Tappy-kw_remove_extreme','%s'%x.remove_extreme)
    f.setncattr('Tappy-kw_zero_ts','%s'%x.zero_ts)
    f.setncattr('Tappy-kw_filter','%s'%x.filter)
    f.setncattr('Tappy-kw_include_inferred','%s'%x.include_inferred)
    
    f.close()

    print 'Wrote ncfile='+ncfile

    # Quick control of resulting output:
    """
    STATION=CUXHAVEN
    start=0
    end=4464 # 31 days
    year='2022'
    start1=52560 # 16 dec of the last model year
    end1=$(echo $start1+$end | bc -l)

    tides_file=$ncfile
    elev_file=$filein # Model file of the last year
    
    ncks -O -d time,$start1,$end1 $elev_file Elev_${STATION}_ncks_$year.nc
    
    ncks -O -d time,$start,$end $tides_file Tides_${STATION}_ncks_$year.nc
    ncks -A -v elev Elev_${STATION}_ncks_$year.nc Tides_${STATION}_ncks_$year.nc
    # Subtract the tides to yield the surge
    ncap2 -O -s "surge=elev-tides" Tides_${STATION}_ncks_$year.nc Surge_${STATION}_ncks_$year.nc
    ncatted -a long_name,surge,m,c,"surge=elev-tides at ${STATION}" Surge_${STATION}_ncks_$year.nc
    ncview Surge_${STATION}_ncks_$year.nc
    """

def read_timeseries( year, STATION, delta_time, num_years, verbose=1):
    filedir, filename = filein_set( year, STATION)
    if verbose > 0:
        print 'Read data from file(s) filein=$filedir/'+filename,
        if num_years > 0:
            print '(etc.)'
        else:
            print ''
        print 'in filedir='+filedir
    
    if regs['seriesfile'] == filename:
        print 'Re-use data already loaded from files', filename
        return None
    series = Dataset(filedir + filename)
    
    try:
        assert(series.variables['elev'][0] > -20.)
    except:
        print 'Masked data in', filename
        return [], [], -1, ''
    
    regs['seriesfile'] = filename
        
    elev0 = series.variables['elev']
    time0 = series.variables['time']
    time_units = time0.units
    if regs['times_def'][1]:
        assert regs['times_def'][1][:4] == time_units[:4] # e.g. 'seco'[nds]
    delta_time = time0[1]-time0[0]
    if regs['times_def'][2]>0:
        assert regs['times_def'][2] == delta_time
    
    if num_years == 0:
        # Analysis within this year
        regs['times_def'] = [ time0, time_units, delta_time ]
        return time0, elev0, len(time0), time_units
    
    time = np.zeros( len(time0) + \
                     (num_years-1)*366*int( round(86400/delta_time)) )
    elev = np.zeros(len(time),dtype=np.float32)
    
    time[:len(time0)] = time0[:]
    elev[:len(time0)] = elev0[:]

    # Concatenate series for the other years
    tindx0 = 0
    year0 = year
    for year in range(year0+1,year0+num_years):
        filedir, filename = filein_set( year, STATION)
        series = Dataset(filedir + filename)
        time0 = series.variables['time']
        assert time_units[:4] == time0.units[:4]
        
        dtime = num2date(time0[0],time0.units) - num2date(time[0],time_units)
        dtime = dtime.days * 86400 + dtime.seconds
        
        tindx0 = int( round((time0[0]+dtime - time[0])/delta_time) )
        assert np.all(time[tindx0:tindx0+10] == time0[:10]+dtime) # no leapsecond
        time[tindx0:tindx0+len(time0)] = time0[:]+dtime
        assert np.all(elev[tindx0:tindx0+10] \
                      == series.variables['elev'][:10] ) # same simulation
        elev[tindx0:tindx0+len(time0)] = series.variables['elev'][:]

    regs['times_def'] = [ time, time_units, delta_time ]
    tindx0+=len(time0)
    
    return time[:tindx0], elev[:tindx0], tindx0, time_units

def print_speed_list(x,HARMall,HARMminor=[]):
    speed_list=[]
    for harm in HARMall:
        speed_list+=[x.tidal_dict[harm]['ospeed']]
    speed_list = np.asarray(speed_list)
    
    speed_argsort = np.argsort(speed_list)
    HARM_sorted = [ HARMall[ii] for ii in speed_argsort ]
    speed_sorted = np.rad2deg(speed_list[speed_argsort])
    speed_dists =  speed_sorted[1:]-speed_sorted[:-1]
    harm=HARM_sorted[0]
    print ''
    print 'Harmonics phase, delta-phase [*:minor]'
    print '    %4s %5.2f' % (harm,speed_sorted[0])
    for ii in range(1,len(HARM_sorted)):
        harm=HARM_sorted[ii]
        if harm in HARMminor:
            print '   *',
        else:
            print '    ',
        print '%4s %5.2f %5.3f' % (harm,speed_sorted[ii],speed_dists[ii-1])

    print '360 deg / 1 month =', 360. / (31.*24.) ,'deg / hour'

# From pwcazenave/tappy readme.md:
# "Set up the bits needed for TAPPY. This is mostly lifted from
#  tappy.py in the baker function "analysis" (around line 1721)."
# cha at FCOO: These are the keywords for analysis, only. They are not relevant
# for prediction, but we will present some of the keywords in the prediction
# netCDF file.
quiet = True
debug = False
outputts = False
outputxml = False
ephemeris = False
rayleigh = 1.0
print_vau_table = False
missing_data = 'ignore'
linear_trend = False
remove_extreme = False
zero_ts = None
filter = None
pad_filters = None
# cha at FCOO: Since this is analysis of a model with boundary conditions a
#              subset of harmonics, we cannot infer any harmonics.
include_inferred = False

if rayleigh > 0:
    ray = float(rayleigh)

x = tappy.tappy(
    outputts = outputts,
    outputxml = outputxml,
    quiet=quiet,
    debug=debug,
    ephemeris=ephemeris,
    rayleigh=rayleigh,
    print_vau_table=print_vau_table,
    missing_data=missing_data,
    linear_trend=linear_trend,
    remove_extreme=remove_extreme,
    zero_ts=zero_ts,
    filter=filter,
    pad_filters=pad_filters,
    include_inferred=include_inferred,
    )

# Script parameters and dicts:

class Astro:
    pass

asc = Astro()
# asc.package[year0] : time subrange of each of x.zeta, x.nu, ...
asc.package = dict()
asc.package['dates'] = dict()
asc.packagefil = ''

def package_from_file(datesid,datestamp=None):

    # Read the saved astronomics factors of all the dates
    
    asc.packagefil = ''
    try:
        # Re-use earlier data (e.g. from the first STATION)
        assert datesid in asc.package.keys()
    except:
        if datestamp is None: datestamp = datesid
        package_pkl = astrodir + 'astronomics_' + datestamp +'.pkl'
        try:
            # assert False
            with open(package_pkl,'r') as packagefile:
                asc.package = pickle.load(packagefile)
        except:
            if not os.path.exists(astrodir): os.makedirs(astrodir)
            asc.packagefil = 'New'
            asc.package = dict()
            asc.package['filename']=package_pkl
    else:
        print 'Read astronomics from file', package_pkl
    
    if not 'dates' in asc.package.keys():
        asc.package['dates'] = dict()
  
def package_to_file():

    # Save the astronomics factors of all the dates to file
    
    if asc.packagefil is not 'New': return
    print 'Save new/updated astronomics package to file', asc.package['filename']
    with open(asc.package['filename'],'w') as packagefil:
        pickle.dump(asc.package,packagefil)
  
def tappy_fill(u, dates, datesid, start=0, end=0, verbose = 1):

    # Fill the tappy instance with the atronomics factors for all dates
    
    if end == 0: end = len(dates)
    u.dates = dates[start:end]
    dates_len = len(u.dates)
    asc.packagefil = ''
    if not datesid in asc.package.keys():
        if verbose > 0:
            print 'Calculate astronomic factors for',datesid
            print 'This takes around a minute (on modeldev02) ...',
        
        sys.stdout.flush()
        
        asc.package[datesid] = u.astronomic(dates)
        asc.packagefil = 'New' # Write a new or updated pickle file at end
        
    if not datesid in asc.package['dates'].keys():
        asc.package['dates'][datesid] = dates[:]
        asc.packagefil = 'New' # Write a new or updated pickle file at end

    package_to_file()
    
    assert asc.package['dates'][datesid][0] == dates[0]
    assert end <= len(asc.package['dates'][datesid])
    package = [[]] * len(asc.package[datesid])
    package[8] = 0
    for ii in range(len(package)):
        if ii == 8: continue
        package[ii] = asc.package[datesid][ii][start:end]    
    
    jd = u.dates2jd([u.dates[0]]) # Julian day
    date_o = np.deg2rad( 360. * (jd[0] - 2400000.5) )
    try:
        assert package[8] == date_o
    except:
        if verbose > 1: print 'update origo'
        package[8] = date_o

    (u.zeta, u.nu, u.nup, u.nupp, u.kap_p, u.ii, u.R, u.Q, u.T, \
     u.jd, u.s, u.h, u.N, u.p, u.p1) = package
    
    # u.dates *is needed* for this routine:
    # Tim Cera comment:
    # "Should change this - runs ONLY to get tidal_dict filled in..."
    (u.speed_dict, u.key_list) = \
                u.which_constituents(dates_len, package, rayleigh_comp=1)
                   
    return u.key_list
            
""" Main script """

ncfile = ''
num_years0 = num_years
years_predict0 = years_predict[:]

stii = 0
sti0 = 0
do_analyze = ''
verb = 2
while stii < len(STATIONS):
    STATION = STATIONS[stii]
    stii+=1
    
    num_years = num_years0
    years_predict = years_predict0
    
    if years_predict:
        # output file
        ncfile = outdir + outfile_templ.replace('[STATION]',STATION)
        
        if os.path.exists(ncfile):
            print 'Will NOT update existing ncfile='+ncfile
            # Shift sti0 if it is the index of the first station to predict:
            if stii == sti0+1: sti0+=1
            continue
            
    if len(years_predict) > 0:
        packagefile_id = str(years_predict[0])+'-'+str(years_predict[-1])
    else:
        packagefile_id = str(years_analyze[0])+'-'+str(years_analyze[-1])
        
    # Cycle over more stations only for a fixed length of num_years
    if not STATION in r_ph[setup_anal].keys():
        if do_analyze == STATION:
            print 'Constituents were not calculated'
            continue
        num_years = len(years_analyze)
        print '%s: Analyze for constituents based on the %d years long period.'%\
              (STATION, num_years),

        if years_predict0: 
            stii-=1
            print 'The prediction is run thereafter for the station'
        do_analyze = STATION

    # Downgrade the verboseness
    if verb > 1 and stii > sti0+1: verb=1
    
    if num_years > 0:
        years_predict = []
        
    # Read the Tappy astronomics from file
    package_from_file(packagefile_id)
    
    if len(years_predict) > 0:

        harm = r_ph[setup_anal][STATION]['harm_sorted']        
        assert set(harm) == set(HARMall)

        rr = dict()
        ph = dict()
        for ii in range(len(harm)):
            rr_ph = r_ph[setup_anal][STATION][harm[ii]]
            rr[harm[ii]] = rr_ph[0]
            ph[harm[ii]] = rr_ph[1]
        
        u = tappy.Util(rr, ph)

        canon_harms = tappy_fill(u,pred_dates,pred_years_id)

        calcdates = np.array(range(len(pred_dates)), dtype=np.float64 ) \
                    * delta_time / 3600. # timeseries step in units of hour
        
        print 'prediction = sum of signals ...'
        sys.stdout.flush()
        
        pred_all = u.sum_signals(HARMall, calcdates, u.tidal_dict)

        # Output to netCDF
        pred_to_netCDF(STATION, ncfile, x, harm, rr, ph, pred_times, pred_all)
        
        
    if len(years_predict) > 0: continue
    
    year = years_analyze[0]
    syear = str(year)
    timeseries = read_timeseries(year, STATION, delta_time, num_years, verbose=verb)
    assert timeseries is not None # Cannot have repeated usage here
    times, elev, series_len, time_units = timeseries
    dates = num2date(times[:],time_units)

    # Analyze for num_years full years
    start = 0   
    end = int(np.rint( num_years * 365.25 * 86400. / delta_time ))
    
    package_id = '%d-%d' %(years_analyze[0],years_analyze[-1])
  
    if verb > 0: print dates[start],
            
    x.elevation = elev[start:end]  
    x.dates = dates[start:end]
    
    assert 'seconds' in time_units
  
    canon_harms = tappy_fill(x, dates, package_id, start=start, end=end, \
                             verbose=verb)
    
    if verb > 1:
        print_speed_list(x,HARMall,HARMminor=HARMminor)
    
    x.key_list = HARMall # Include them all
        
    x.speed_dict = dict()
    for harm in x.key_list:
        x.speed_dict[harm] = x.tidal_dict[harm]
    
    if verb > 1:
        print 'Analysis for',  x.key_list, '...'
        print 'This takes more than 2 minutes (on modeldev02).'
        print ''
        sys.stdout.flush()
    elif verb > 0:
        print 'Analysis ...'
        sys.stdout.flush()
    
    x.constituents() # the analysis
    
    indxs = np.argsort( [x.r[key] for key in x.key_list] )[::-1]
    harm_sorted = [ x.key_list[indx] for indx in indxs]

    if not setup_anal in r_ph.keys():
        r_ph[setup_anal] = dict()
    if not STATION in r_ph[setup_anal].keys():
        r_ph[setup_anal][STATION] = dict()
    for harm in x.key_list:
        r_ph[setup_anal][STATION][harm] = [ x.r[harm], x.phase[harm] ]
    r_ph[setup_anal][STATION]['harm_sorted'] = harm_sorted
    with open(harmonics_pkl,'w') as stfil:
         pickle.dump(r_ph, stfil)
    print 'Dumped r_ph to', harmonics_pkl
    
    if verb > 1:
        cum_var = np.sqrt(np.cumsum(\
                        [x.r[harm]**2 for harm in harm_sorted[::-1]] )[::-1])
    
        for jj in range(len(indxs)):
            indx = indxs[jj]
            harm = x.key_list[indx]
            print( '{:<2} {:>5} : {:>7.2f} {:>7.4f} {:>7.4f}'.format(\
                    indx,harm,x.phase[harm],x.r[harm],cum_var[jj]) )


# End of loop over STATIONS

# END OF SCRIPT

"""
To get STATIONS list from model output directory

setup_run='...'
filedir='...'
cd $filedir/${setup_run}/cat_2021

stations='['
for file in ST_*_2Dvars_10min*; do
    if [[ "$(echo $file | grep 'PILOT_')" ]]; then continue; fi
    if [[ "$(echo $file | grep 'NOVANA_')" ]]; then continue; fi
    if [[ "$(echo $file | grep 'FRV_')" ]]; then continue; fi
    if [[ "$(echo $file | grep 'SMHI_')" ]]; then continue; fi
    if [[ "$(echo $file | grep 'IOW_')" ]]; then continue; fi
    if [[ "$(echo $file | grep 'SYKE_')" ]]; then continue; fi
    if [[ "$(echo $file | grep 'DMU_')" ]]; then continue; fi
    ST=${file/_2Dvars*/}
    stations+='"'${ST/ST_/}'",'
done
stations+=']'
echo STATIONS=$stations

stations='['
for file in ST_*_2Dvars_10min*; do
    if [[ "$(echo $file | grep '\(PILOT_\|NOVANA_\|FRV_\|SMHI_\|IOW_\|SYKE_\|DMU_\)')" ]]
    then
        ST=${file/_2Dvars*/}
        stations+='"'${ST/ST_/}'",'
    fi
done
stations+=']'
insts_stations=$stations
echo STATIONS=$insts_stations
"""


"""
References:

R. D. Ray, On Tidal Inference in the Diurnal Band, 2017 
Journal of Atmospheric and Oceanic Technology Volume 34, Issue 2
DOI:    https://doi.org/10.1175/JTECH-D-16-0142.1
(Cite: "For some altimeter-constrained tide models, an inferred P1 constituent is found to be more accurate than a directly determined one.")


Seasonal variation of the principal tidal constituentsin the Bohai Sea
Daosheng Wang et al., Ocean Sci., 16, 114, 2020
DOI: https://doi.org/10.5194/os-16-1-2020
(Available from: https://www.researchgate.net/publication/338404647_Seasonal_variation_of_the_principal_tidal_constituents_in_the_Bohai_Sea [accessed Apr 06 2022])
(Cite: "A recent work on seasonal variations, based on a method of 'enhanced harmonic analysis' (Jin et al., 2018) that provides temporally varying mean sea level and tidal harmonic parameters during each of a series of (monthly) time segments where a 'segmented harmonic analysis' is performed (as introduced originally by Foreman et al., 1995). '[...] nodal and astronomical argument corrections are embedded into the least square fit, following Foreman et al. (2009); in addition, theharmonic parameters of the minor tidal constituents are assumed to be constant and calculated together with the temporally varying harmonic parameters of the principal tidal constituent'")


Wenguo Li, Bernhard Mayera, Thomas Pohlman: The influence of baroclinity on tidal ranges in the North Sea, December 2020, Estuarine Coastal and Shelf Science 250
DOI: https://doi.org/10.1016/j.ecss.2020.107126

(Cite: "Long (65 years) timeseries measures are constructed for what they call 'the spatial distribution and temporal variations of tidal range difference (TRD) induced by the baroclinity'. Conclusion: '[...] absolute summer TRD is much bigger than the winter one.'")

Mawdsley RJ and Haigh ID (2016). Spatial and Temporal Variability and Long-Term Trends in Skew Surges Globally. Front. Mar. Sci. 3:29.doi: 10.3389/fmars.2016.00029 
Available from: https://www.researchgate.net/publication/299279604_Spatial_and_Temporal_Variability_and_Long-Term_Trends_in_Skew_Surges_Globally [accessed Apr 06 2022]

(Cite: Compare usage of 'non-tidal residual (NTR)' versus 'skew surge' in investigation of 'extreme sea levels (ESL)'. 'A skew surge is the differencebetween the maximum observed sea level and the maximumpredicted tidal level regardless of their timing during the tidal cycle. "The tide-surge interaction is strongest in regions of shallow bathymetry such as the North Sea, north Australia and the Malay Peninsula.")

Adam Devlin et al., 2018: Seasonality of Tides in Southeast Asian Waters. Journal of Physical Oceanography 48(5)
DOI: 10.1175/JPO-D-17-0119.1

(Cite: "Harmonic analysis of the hourly sea level observations Zobs(t) is performed using the R_T_Tide package (Pawlowicz et al. 2002; Leffler and Jay 2009)")



"""
