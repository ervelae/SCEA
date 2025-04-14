
import os

import netCDF4
import numpy as np
import pandas
import xarray


def process_filename(filename):
    """
    Extract satellite name, orbit number, timestamp and equator starting
    longitude from filename. Sentinel-7 is changed to CO2M.
    """
    filename = os.path.basename(filename)
    filename = os.path.splitext(filename)[0]

    if filename.startswith('cosmo_2d'):
        sat = None
        orbit = None
        date = pandas.to_datetime(filename, format='cosmo_2d_%Y%m%d%H')
        lon = None
    else:
        filename = filename.split('_')

        if filename[1] == '5':
            sat, n, date, orbit, lon = filename[:5]
            sat = '_'.join([sat, n])
        elif filename[0] == 'CO2M':
            sat, date, orbit, lon = filename
        else:
            sat, n, t, date, orbit, lon = filename[:6]
            sat = '_'.join([sat, n, t])

        orbit = int(orbit[1:])
        date = pandas.to_datetime(date, format='%Y%m%d%H')
        lon = int(lon[1:])

    return sat, orbit, date, lon



def read_fields(filename, tracers, time=None, correct_berlin=True,
        slice_=Ellipsis, use_xarray=False):
    """\
    Read fields from netCDF file.
    
    If `correct_berlin` is True, correct '*_BV' tracers for January and
    July by scaling '*_B0' with 0.55 which corrects the too high emissions
    in January and July in the SMARTCARB simulations. This is done to fix a
    bug in the first month of simulations where large point sources were
    added twice to the emission field.
    """
    with netCDF4.Dataset(filename) as nc:
        fields = []

        for tracer in tracers:
            if correct_berlin and tracer.endswith('_BV'):

                field = nc.variables[tracer]
                attrs = dict((k, field.getncattr(k)) for k in field.ncattrs())

                label = tracer.split('_')[0]

                berlin_tracers = ['%s_B%d' % (label, i) for i in range(3)]
                b0,b1,b2 = read_fields(filename, berlin_tracers, time,
                        correct_berlin, slice_, use_xarray=use_xarray)

                if time.month in [1,7]:
                    field = 0.55 * b0 + b1 + b2
                else:
                    field = b0 + b1 + b2

                if use_xarray:
                    field.attrs.update(attrs)

            else:
                field = nc.variables[tracer]

                if use_xarray:
                    attrs = dict((k, field.getncattr(k)) for k in field.ncattrs())
                    field = xarray.DataArray(field[:], dims=field.dimensions, attrs=attrs)

                if slice_ is Ellipsis or isinstance(slice_, slice):
                    field = field[slice_]
                else:
                    # use slice starting at last dimension (instead of first)
                    ndim = np.ndim(field)

                    if len(slice_) == ndim:
                        field = field[slice_]
                    else:
                        s = [Ellipsis] * (ndim - len(slice_)) + list(slice_)
                        field = field[s]


            # correct _BC fields
            if correct_berlin and tracer.endswith('_BC') and time.month in [1,7]:
                if tracer.startswith('NO2'):
                    field *= 0.8038946
                elif tracer.startswith('XCO2') or tracer.startswith('YCO2'):
                    field *= 0.7078081
                else:
                    raise ValueError


            field = np.squeeze(field)
            fields.append(field)

    return fields


def read_trace_gas_field(filename, trace_gas, time=None,
                         correct_berlin=True, clip_data=True,
                         slice_=Ellipsis, use_constant_emissions=False,
                         use_xarray=False, scaling=None):
    """\
    Read total field of trace gases (CO2, NO2 or CO).
    """
    if scaling is None:
        scaling = 1.0

    suffix = '_BC' if use_constant_emissions else '_BV'

    if trace_gas in ['CO2', 'XCO2']:
        fields = ['XCO2%s' % suffix, 'XCO2_A', 'XCO2_JV', 'XCO2_RA', 'XCO2_GPP', 'XCO2_BG']
        bv,a,jv,ra,gpp,bg = read_fields(filename, fields, time, correct_berlin,
                                        slice_, use_xarray=use_xarray)

        field = scaling * (bv + a + jv) + ra - gpp + bg

        if use_xarray:
            attrs = {
                'standard name': 'CO2_column-averaged_dry-air_mole_fraction',
                'long name': 'CO2_column-averaged_dry-air_mole_fraction',
                'units': 'ppm'
            }
            field.attrs.update(attrs)

        if clip_data:
            field[:] = np.where((field < 0) | (field > 1e5), np.nan, field)

    else:
        if trace_gas == 'NO2':
            fields = ['NO2%s' % suffix, 'NO2_A', 'NO2_JV', 'NO2_BG']
            bv,a,jv,bg = read_fields(filename, fields, time, correct_berlin,
                                     slice_, use_xarray=use_xarray)
            field = scaling * (bv + a + jv) + bg

        if trace_gas == 'CO':
            fields = ['CO%s' % suffix, 'CO_A', 'CO_JV', 'CO_BG']
            bv,a,jv,bg = read_fields(filename, fields, time, correct_berlin,
                                     slice_, use_xarray=use_xarray)
            field = scaling * (bv + a + jv) + bg

        if use_xarray:
            attrs = {
                'standard name': '%s_vertical_column_density' % trace_gas,
                'long name': '%s_vertical_column_density' % trace_gas,
                'units': 'molecules cm-2'
            }
            field.attrs.update(attrs)
            
    return field



def read_trace_gas_noise(filename, tracer, use_epf=True, level=0.7, slice_=Ellipsis,
                      seed='orbit', use_xarray=False):
    """
    tracer in {'CO2', 'NO2', 'CO', 'uXCO2', 'uNO2_low', 'uNO2_high',
               'uCO_low', 'uCO_high'}
    level in {0.5, 0.7, 1.0, 'low', 'high'}
    """
    sat,orbit,_,lon_eq = process_filename(filename)

    

    if sat == 'Sentinel_5':
        utracer = 'uNO2'
    else:
        if tracer == 'CO2':
            utracer = 'uXCO2'
        elif tracer == 'NO2':
            utracer = 'uNO2_%s' % level
        else:
            utracer = 'uCO_%s' % level

    # read uncertainty (TODO/FIXME: correct uNO2 for Jan/Jul)
    u = read_fields(filename, [utracer], slice_=slice_, use_xarray=use_xarray)[0]

    if not use_xarray:
        u[u.mask] = np.nan
        u = u.data

    # seed based on orbit and lon_eq for same noise patterns with each species
    if seed == 'orbit':
        offset = {
            'CO2':         0,
            'NO2': 300000000,
            'CO':  600000000
        }[tracer]
        np.random.seed(10000 * lon_eq + orbit + offset)
    else:
        np.random.seed(seed)

    noise = np.random.randn(*u.shape).astype(u.dtype)

    if tracer == 'CO2':
        
        level = {'low': 0.5, 'medium': 0.7, 'high': 1.0}[level]
        
        if use_epf:
            noise = level * noise * u / 1.50
        else:
            noise = level * noise

    #elif tracer == 'NO2' and use_epf:
    #    cc = read_fields(filename, ['CLCT'], slice_=slice_)[0]
    #
    #    # use Wenig et al. (2008) with `uNO2` instead of 1.5e15
    #    noise = xarray.DataArray((1.0 + 3.0 * cc) * u * noise, dims=u.dims)

    # MODOFIED
    elif tracer == 'NO2' and use_epf:
        
        #cc = read_fields(filename, ['CLCT'], slice_=slice_)[0]
        cc = 0  #cloud cover

        # use Wenig et al. (2008) with `uNO2` instead of 1.5e15
        noise = xarray.DataArray((1.0 + 3.0 * cc) * u * noise, dims=u.dims)

    else:
        noise = u * noise


    if use_xarray:
        if tracer == 'CO2':
            name = '%s random noise (using %s and %.1f ppm)' % (tracer, utracer, level)
        else:
            name = '%s random noise (using %s)' % (tracer, utracer)

        noise.attrs['standard name'] = name
        noise.attrs['long name'] = name
        noise.attrs['units'] = 'ppm' if tracer == 'CO2' else 'molecules cm-2'

    return noise



def read_cosmo(filename, trace_gas):
    
    time = process_filename(filename)[2]
    
    data = {}
    
    for name in ['rlon', 'rlat', 'lon', 'lat', 'CLCT', 'PS']:
        data[name] = read_fields(filename, [name], use_xarray=True)[0]
        
    data[trace_gas]= read_trace_gas_field(filename, trace_gas, time=time, use_xarray=True)
    
    return xarray.Dataset(data)
    
    
def read_level2(filename,
                co2_noise_scenario='medium', co2_cloud_threshold=0.01, co2_scaling=1.0,
                no2_noise_scenario='medium', no2_cloud_threshold=0.30, no2_scaling=1.0,
                co_noise_scenario=None, co_cloud_threshold=0.05, co_scaling=1.0,
                use_constant=False, seed='orbit', 
                only_observations=True, add_background=False):
    """
    Read synthetic XCO2, NO2 and CO observations from SMARTCARB project.
    
    
    filename
    
    co2_noise: 0.5, 0.7 or 1.0 ppm for low-, medium- or high-noise scenario.
    co2_thr
    
    """
    data = {}
    dims = ('along', 'across')
    dims2 = ('along', 'across', 'corners')

    satellite, orbit, time, lon_eq = process_filename(filename)
    
    if satellite == 'Sentinel_7_CO2':
        satellite = 'CO2M'
            
    data['time'] = pandas.Timestamp(time)

    for name1, name2 in [
        ('lon', 'longitude'),
        ('lat', 'latitude'),
        ('lonc', 'longitude_corners'),
        ('latc', 'latitude_corners'),
        ('clouds', 'CLCT'),
        ('psurf', 'PS')
    ]:
        data[name1] = read_fields(filename, [name2], use_xarray=True)[0]
     
 
    if add_background:
        bg, resp, gpp = read_fields(filename, ['XCO2_BG', 'XCO2_RA', 'XCO2_GPP'], use_xarray=True)
        data['CO2_BG'] = bg + resp - gpp
        data['NO2_BG'] = read_fields(filename, ['NO2_BG'], use_xarray=True)[0]
        data['CO_BG'] = read_fields(filename, ['CO_BG'], use_xarray=True)[0]
        

    # CO2 and NO2
    for trace_gas, noise, thr, scaling in [
        ('CO2', co2_noise_scenario, co2_cloud_threshold, co2_scaling),
        ('NO2', no2_noise_scenario, no2_cloud_threshold, no2_scaling),
        ('CO', co_noise_scenario, co_cloud_threshold, co_scaling)
    ]:
        if noise is None or thr is None:
            continue

        attrs = {'noise_level': noise, 'cloud_threshold': thr}

        val = read_trace_gas_field(filename, trace_gas, time=time,
                                   use_constant_emissions=use_constant,
                                   use_xarray=True, scaling=scaling)

        err = read_trace_gas_noise(filename, trace_gas, level=noise,
                                   use_epf=True, seed=seed, use_xarray=True)

        is_cloudy = data['clouds'] > thr

        # store noisefree/gapfree data
        if not only_observations:
            data[f'{trace_gas}_noisefree'] = val.copy()

        val[:] = np.where(is_cloudy, np.nan, val)
        err[:] = np.where(is_cloudy, np.nan, err)

        data[trace_gas] = err + val
        
        data[f'{trace_gas}_std'] = xarray.zeros_like(val)
        
        if trace_gas == 'NO2':
            data[f'{trace_gas}_std'][:] = {'low': 1e15, 'medium': 1.5e15, 'high': 2e15}[noise]
        elif trace_gas == 'CO':
            data[f'{trace_gas}_std'][:] = {'low': 4e17, 'high': 4e17}[noise]
        else:
            data[f'{trace_gas}_std'][:] = {'low': 0.5, 'medium': 0.7, 'high': 1.0}[noise]
        
        if not only_observations:
            data[f'{trace_gas}_noise'] = err

        data[trace_gas].attrs.update(attrs)
        data[trace_gas].attrs.update(val.attrs)
        data[trace_gas].attrs['standard name'] += ' with random noise'
        data[trace_gas].attrs['long name'] += ' with random noise'
        
        if scaling is None:
            data[trace_gas].attrs['scaling'] = 1.0
        else:
            data[trace_gas].attrs['scaling'] = scaling

            
    return xarray.Dataset(data, attrs={'satellite': satellite, 'orbit': orbit,
                                       'lon_eq': lon_eq})