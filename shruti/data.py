""" File for handling data loading and saving. """
import os
import glob
import time
import datetime
import pickle
import numpy as np

#from kerchunk.zarr import ZarrToZarr
#from kerchunk.combine import MultiZarrToZarr
import numpy as np
import netCDF4 as nc
import xarray as xr
import xesmf
import h5py

import read_config


data_paths = read_config.get_data_paths()
TRUTH_PATH = data_paths["GENERAL"]["TRUTH_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]

#all_fcst_fields = ['cape', 'cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp', 'u700', 'v700']
all_fcst_fields = {
    "Convective available potential energy":"cape",
    "Convective precipitation (water)":"acpcp",#    "Medium cloud cover",
    "Surface pressure":"sp",
    "Upward short-wave radiation flux":"uswrf",
    "Downward short-wave radiation flux":"dswrf",
    "2 metre temperature":"t2m",
    "Cloud water":"cwat",
    "Precipitable water":"pwat",
    "Ice water mixing ratio":"icmr",
    "Cloud mixing ratio":"clwmr",
    "Rain mixing ratio":"rwmr",
    "Total Precipitation":"tp",
    "U component of wind":"u",
    "V component of wind":"v",
}

all_fcst_levels = {
    "Convective available potential energy": "surface",
    "Convective precipitation (water)": "surface",
    "Medium cloud cover": "middleCloudLayer",
    "Surface pressure": "surface",
    "Upward short-wave radiation flux": "surface",
    "Downward short-wave radiation flux": "surface",
    "2 metre temperature": "heightAboveGround",
    "Cloud water": "atmosphereSingleLayer",
   "Precipitable water": "atmosphereSingleLayer",
    "Ice water mixing ratio": "isobaricInhPa",
    "Cloud mixing ratio": "isobaricInhPa",
    "Rain mixing ratio": "isobaricInhPa",
    "Total Precipitation": "surface",
    "U component of wind": "isobaricInhPa",
    "V component of wind": "isobaricInhPa",
}


accumulated_fields = ["Convective precipitation (water)", "ssr", "Total Precipitation"]
nonnegative_fields = [
    "Convective available potential energy",
    "Convective precipitation (water)",
    "Medium cloud cover",
    "Surface pressure",
    "Upward short-wave radiation flux",
    "Downward short-wave radiation flux",
    "2 metre temperature",
    "Cloud water",
    "Precipitable water",
    "Ice water mixing ratio",
    "Cloud mixing ratio",
    "Rain mixing ratio",
    "Total Precipitation",
]

HOURS = 6  # 6-hr data modified to 24 hour

lat_reg_b = (
    np.array(
        [
            25.25,
            25.0,
            24.75,
            24.5,
            24.25,
            24.0,
            23.75,
            23.5,
            23.25,
            23.0,
            22.75,
            22.5,
            22.25,
            22.0,
            21.75,
            21.5,
            21.25,
            21.0,
            20.75,
            20.5,
            20.25,
            20.0,
            19.75,
            19.5,
            19.25,
            19.0,
            18.75,
            18.5,
            18.25,
            18.0,
            17.75,
            17.5,
            17.25,
            17.0,
            16.75,
            16.5,
            16.25,
            16.0,
            15.75,
            15.5,
            15.25,
            15.0,
            14.75,
            14.5,
            14.25,
            14.0,
            13.75,
            13.5,
            13.25,
            13.0,
            12.75,
            12.5,
            12.25,
            12.0,
            11.75,
            11.5,
            11.25,
            11.0,
            10.75,
            10.5,
            10.25,
            10.0,
            9.75,
            9.5,
            9.25,
            9.0,
            8.75,
            8.5,
            8.25,
            8.0,
            7.75,
            7.5,
            7.25,
            7.0,
            6.75,
            6.5,
            6.25,
            6.0,
            5.75,
            5.5,
            5.25,
            5.0,
            4.75,
            4.5,
            4.25,
            4.0,
            3.75,
            3.5,
            3.25,
            3.0,
            2.75,
            2.5,
            2.25,
            2.0,
            1.75,
            1.5,
            1.25,
            1.0,
            0.75,
            0.5,
            0.25,
            0.0,
            -0.25,
            -0.5,
            -0.75,
            -1.0,
            -1.25,
            -1.5,
            -1.75,
            -2.0,
            -2.25,
            -2.5,
            -2.75,
            -3.0,
            -3.25,
            -3.5,
            -3.75,
            -4.0,
            -4.25,
            -4.5,
            -4.75,
            -5.0,
            -5.25,
            -5.5,
            -5.75,
            -6.0,
            -6.25,
            -6.5,
            -6.75,
            -7.0,
            -7.25,
            -7.5,
            -7.75,
            -8.0,
            -8.25,
            -8.5,
            -8.75,
            -9.0,
            -9.25,
            -9.5,
            -9.75,
            -10.0,
            -10.25,
            -10.5,
            -10.75,
            -11.0,
            -11.25,
            -11.5,
            -11.75,
            -12.0,
            -12.25,
            -12.5,
            -12.75,
            -13.0,
            -13.25,
            -13.5,
            -13.75,
            -14.0,
        ]
    )
    - 0.125
)
lat_reg = 0.5 * (lat_reg_b[1:] + lat_reg_b[:-1])

lon_reg_b = (
    np.array(
        [
            19.0,
            19.25,
            19.5,
            19.75,
            20.0,
            20.25,
            20.5,
            20.75,
            21.0,
            21.25,
            21.5,
            21.75,
            22.0,
            22.25,
            22.5,
            22.75,
            23.0,
            23.25,
            23.5,
            23.75,
            24.0,
            24.25,
            24.5,
            24.75,
            25.0,
            25.25,
            25.5,
            25.75,
            26.0,
            26.25,
            26.5,
            26.75,
            27.0,
            27.25,
            27.5,
            27.75,
            28.0,
            28.25,
            28.5,
            28.75,
            29.0,
            29.25,
            29.5,
            29.75,
            30.0,
            30.25,
            30.5,
            30.75,
            31.0,
            31.25,
            31.5,
            31.75,
            32.0,
            32.25,
            32.5,
            32.75,
            33.0,
            33.25,
            33.5,
            33.75,
            34.0,
            34.25,
            34.5,
            34.75,
            35.0,
            35.25,
            35.5,
            35.75,
            36.0,
            36.25,
            36.5,
            36.75,
            37.0,
            37.25,
            37.5,
            37.75,
            38.0,
            38.25,
            38.5,
            38.75,
            39.0,
            39.25,
            39.5,
            39.75,
            40.0,
            40.25,
            40.5,
            40.75,
            41.0,
            41.25,
            41.5,
            41.75,
            42.0,
            42.25,
            42.5,
            42.75,
            43.0,
            43.25,
            43.5,
            43.75,
            44.0,
            44.25,
            44.5,
            44.75,
            45.0,
            45.25,
            45.5,
            45.75,
            46.0,
            46.25,
            46.5,
            46.75,
            47.0,
            47.25,
            47.5,
            47.75,
            48.0,
            48.25,
            48.5,
            48.75,
            49.0,
            49.25,
            49.5,
            49.75,
            50.0,
            50.25,
            50.5,
            50.75,
            51.0,
            51.25,
            51.5,
            51.75,
            52.0,
            52.25,
            52.5,
            52.75,
            53.0,
            53.25,
            53.5,
            53.75,
            54.0,
            54.25,
            54.5,
            54.75,
        ]
    )
    - 0.125
)
lon_reg = 0.5 * (lon_reg_b[1:] + lon_reg_b[:-1])

data_path = glob.glob(TRUTH_PATH + "*.nc")

ds = xr.open_mfdataset(data_path[0])
# print(ds)

lat_reg_IMERG = ds.latitude.values
lon_reg_IMERG = ds.longitude.values

lat_reg_IMERG_b = np.append((lat_reg_IMERG - 0.05), lat_reg_IMERG[-1] + 0.05)
# print(lat_reg_IMERG_b)
lon_reg_IMERG_b = np.append((lon_reg_IMERG - 0.05), lon_reg_IMERG[-1] + 0.05)
# print(lon_reg_IMERG_b)
# utility function; generator to iterate over a range of dates


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 100 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1.0, 100.0)


def logprec(y, log_precip=False):
    if log_precip:
        return np.log10(1.0 + y)
    else:
        return y


def get_dates(year, start_hour, end_hour):
    """
    Returns list of valid forecast start dates for which 'truth' data
    exists, given the other input parameters. If truth data is not available
    for certain days/hours, this will not be the full year. Dates are returned
    as a list of YYYYMMDD strings.

    Parameters:
        year (int): forecasts starting in this year
        start_hour (int): Lead time of first forecast desired
        end_hour (int): Lead time of last forecast desired
    """
    # sanity checks for our dataset
    assert year in (2018, 2019, 2020, 2021, 2022)
    assert start_hour >= 0
    assert end_hour <= 168
    assert start_hour % HOURS == 0
    assert end_hour % HOURS == 0
    assert end_hour > start_hour

    # Build "cache" of truth data dates/times that exist
    truth_cache = set()
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(
        year + 1, 1, end_hour // 24 + 2
    )  # go a bit into following year
    #print(start_date, end_date)
    for curdate in daterange(start_date, end_date):
        datestr = curdate.strftime("%Y%m%d")
        
        datestr_true = (curdate + datetime.timedelta(days=1)).strftime("%Y%m%d")
        for hr in [6]:
            fname = f"{datestr_true}_{hr:02}"
            # print(os.path.join(FCST_PATH, test_file))
            if os.path.exists(
                os.path.join(TRUTH_PATH, f"{fname}.nc")):
                truth_cache.add(fname)

    # Now work out which IFS start dates to use. For each candidate start date,
    # work out which truth dates+times are needed, and check if they exist.
    # start_date = datetime.date(year, 1, 1)
    # end_date = datetime.date(year+1, 1, 1)
    valid_dates = []

    for curdate in daterange(start_date, end_date):
        # Check interval by interval.  Not particularly efficient, but almost
        # certainly not a bottleneck, since we're just repeatedly testing
        # inclusion in a Python set, and never hitting the disk
        valid = True

        for hr in [24]:  # range(start_hour, end_hour, HOURS)
            # implicitly assumes 00Z forecasts; needs editing for 12Z
            truth_dt = curdate + datetime.timedelta(days=1)
            # this works for our specific naming convention, where e.g.,
            # 20190204_06 contains the truth data for hours 06-12 on date 20190204
            # print(truth_dt.strftime('%Y%m%d_%H'))
            if truth_dt.strftime("%Y%m%d_06") not in truth_cache:
                valid = False
                break
        if valid:
            datestr = curdate.strftime("%Y%m%d")
            # print(datestr)
            valid_dates.append(datestr)

    return valid_dates


def load_truth_and_mask(date, time_idx, log_precip=False, consolidated=False):
    """
    Returns a single (truth, mask) item of data.
    Parameters:
        date: forecast start date
        time_idx: forecast 'valid time' array index
        log_precip: whether to apply log10(1+x) transformation
    """
    # convert date and time_idx to get the correct truth file
    fcst_date = datetime.datetime.strptime(date, "%Y%m%d")
    valid_dt = fcst_date + datetime.timedelta(hours=int(time_idx))  # needs to change for 12Z forecasts
    #valid_dt = fcst_date + datetime.timedelta(days=1)

    if consolidated:

        d = valid_dt
        d_end = d + datetime.timedelta(hours=6)
        
        # A single IMERG data file to get latitude and longitude
        IMERG_data_dir = '/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07/late_run'
        IMERG_file_name = f"{IMERG_data_dir}/2024/Apr/3B-HHR-L.MS.MRG.3IMERG.20240430-S233000-E235959.1410.V06E.HDF5"
        
        # HDF5 in the ICPAC region
        h5_file = h5py.File(IMERG_file_name)
        latitude = h5_file['Grid']["lat"][763:1147]
        longitude = h5_file['Grid']["lon"][1991:2343]
        h5_file.close()
    
        # The 6h average rainfall
        rain_IMERG_6h = np.zeros((len(longitude), len(latitude)))
        
        start_time = time.time()

        while (d<d_end):
            
            # Load an IMERG file with the current date
            d2 = d + datetime.timedelta(seconds=30*60-1)
            
            # Number of minutes since 00:00
            count = int((d - datetime.datetime(d.year, d.month, d.day)).seconds / 60)
            IMERG_file_name = glob.glob(f"{IMERG_data_dir}/{d.year}/{d.strftime('%b')}/3B-HHR-L.MS.MRG.3IMERG.{d.year}{d.month:02d}{d.day:02d}-S{d.hour:02d}{d.minute:02d}00-E{d2.hour:02d}{d2.minute:02d}{d2.second:02d}.{count:04d}.V06*.HDF5")[0]
            
            h5_file = h5py.File(IMERG_file_name)
            times = h5_file['Grid']["time"][:]
            rain_IMERG = h5_file['Grid']["precipitationCal"][0,1991:2343,763:1147]
            h5_file.close()
    
            # Check the time is correct
            if (d != datetime.datetime(1970,1,1) + datetime.timedelta(seconds=int(times[0]))):
                print(f"Incorrect time for {d}")
    
            # Accumulate the rainfall
            rain_IMERG_6h += rain_IMERG
    
            # Move to the next date
            d += datetime.timedelta(minutes=30)
            
        # Normalise to mm/h
        rain_IMERG_6h /= (2*6)
        rain_IMERG_6h = np.moveaxis(rain_IMERG_6h, [0, 1], [1,0])
        #print(rain_IMERG_6h.shape)
        mask = np.full(rain_IMERG_6h.shape, False, dtype=bool)
        if log_precip:
            return np.log10(1 + rain_IMERG_6h), mask
        else:
            return rain_IMERG_6h, mask

    else:
        fname = valid_dt.strftime("%Y%m%d_%H")#valid_dt.strftime("%Y%m%d")
        data_path = glob.glob(TRUTH_PATH + f"{fname}.nc")
        #print(data_path)
        #ds = xr.concat([xr.open_dataset(dataset).expand_dims(dim={'time':i}, axis=0) for i,dataset in enumerate(data_path)],dim='time').mean('time')
        ds = xr.open_dataset(data_path[0])
    
        da = ds["precipitation"]
        y = da.values
        ds.close()
    
        # mask: False for valid truth data, True for invalid truth data
        # (compatible with the NumPy masked array functionality)
        # if all data is valid:
        mask = np.full(y.shape, False, dtype=bool)
    
        if log_precip:
            return np.log10(1 + y), mask
        else:
            return y, mask


def load_hires_constants(batch_size=1):
    oro_path = os.path.join(CONSTANTS_PATH, "elev.nc")
    df = xr.load_dataset(oro_path)
    # Orography in m.  Divide by 10,000 to give O(1) normalisation
    z = df["elevation"].values
    z /= 10000.0
    df.close()

    lsm_path = os.path.join(CONSTANTS_PATH, "lsm.nc")
    df = xr.load_dataset(lsm_path)
    # LSM is already 0:1
    lsm = df["lsm"].values
    df.close()

    temp = np.stack([z, lsm], axis=-1)  # shape H x W x 2
    return np.repeat(
        temp[np.newaxis, ...], batch_size, axis=0
    )  # shape batch_size x H x W x 2


def load_fcst_truth_batch(
    dates_batch,
    time_idx_batch,
    fcst_fields=all_fcst_fields,
    log_precip=False,
    norm=False,
    consolidated=False
):
    """
    Returns a batch of (forecast, truth, mask) data, although usually the batch size is 1
    Parameters:
        dates_batch (iterable of strings): Dates of forecasts
        time_idx_batch (iterable of ints): Corresponding 'valid_time' array indices
        fcst_fields (list of strings): The fields to be used
        log_precip (bool): Whether to apply log10(1+x) transform to precip-related forecast fields, and truth
        norm (bool): Whether to apply normalisation to forecast fields to make O(1)
    """
    batch_x = []  # forecast
    batch_y = []  # truth
    batch_mask = []  # mask

    for time_idx, date in zip(time_idx_batch, dates_batch):
        batch_x_temp = load_fcst_stack(
            fcst_fields, date, time_idx, log_precip=log_precip, norm=norm, consolidated=consolidated
        )
        batch_x_temp[np.isnan(batch_x_temp)] = 0
        batch_x_temp[np.isinf(batch_x_temp)] = 0
        batch_x.append(batch_x_temp)
        truth, mask = load_truth_and_mask(date, time_idx, log_precip=log_precip, consolidated=consolidated)
        batch_y.append(truth)
        batch_mask.append(mask)

    return np.array(batch_x), np.array(batch_y), np.array(batch_mask)


def load_fcst(field, date, time_idx, log_precip=False, norm=False, consolidated=False):
    """
    Returns forecast field data for the given date and time interval.

    Four channels are returned for each field:
        - instantaneous fields: mean and stdev at the start of the interval, mean and stdev at the end of the interval
        - accumulated field: mean and stdev of increment over the interval, and the last two channels are all 0
    """

    yearstr = date[:4]
    year = int(yearstr)
    # ds_path = os.path.join(FCST_PATH, yearstr, f"{field}.nc")
    #file = glob.glob(FCST_PATH+"gfs"+str(date).replace('-','')+"*_f030_f054_"+field.replace(' ','-')+"_"+all_fcst_levels[field]+".zarr")
    #print(file)
    if consolidated:
        date = datetime.datetime.strptime(date, "%Y%m%d")
        date = date + datetime.timedelta(hours=int(time_idx))  # needs to change for 12Z forecasts
    ds_path = os.path.join(FCST_PATH, yearstr, f"{all_fcst_fields[field]}.zarr")
    
    #ds_path = file[0]
    

    # open using netCDF
    # nc_file = nc.Dataset(ds_path, mode="r")

    try:
        nc_file = xr.open_dataset(ds_path, engine="zarr", consolidated=False)
        nc_file = nc_file.sel(
            {"time":date}
        )

    except:

        try:
            nc_file = xr.open_dataset(ds_path.replace(".zarr","_mid.zarr"), engine="zarr", consolidated=False)
            nc_file = nc_file.sel(
                {"time":date}
            )
                
        except:
            nc_file = xr.open_dataset(ds_path.replace(".zarr","_late.zarr"), engine="zarr", consolidated=False)
            nc_file = nc_file.sel(
                {"time":date}
            )

    if consolidated:
        return np.moveaxis(np.squeeze(nc_file.to_dataarray().values),0,-1)       

    if len(nc_file.step.values)>9:
        nc_file = nc_file.sel({'step':nc_file.step.values[::3]})

    if all_fcst_levels[field] == "isobaricInhPa":
        nc_file = nc_file.sel({"isobaricInhPa": 200})

    # all_data_mean = nc_file[f"{field}_mean"]
    # all_data_sd = nc_file[f"{field}_sd"]

    short_names = list(nc_file.variables.keys())

    short_name = [
        short_name
        for short_name in short_names
        if short_name
        not in [
            "latitude",
            "longitude",
            "step",
            all_fcst_levels[field],
            "surface",
            "time",
            "valid_time",
        ]
    ][0]

    idx_x = nc_file[short_name].values.shape[1]
    idx_y = nc_file[short_name].values.shape[2]
    all_data = np.squeeze(nc_file[short_name].values)

    # data is stored as [day of year, valid time index, lat, lon]

    # calculate first index (i.e., day of year, with Jan 1 = 0)
    # fcst_date = datetime.datetime.strptime(date, "%Y%m%d").date()
    # fcst_idx = fcst_date.toordinal() - datetime.date(year, 1, 1).toordinal()

    #if field in accumulated_fields:
    #    # return mean, sd, 0, 0.  zero fields are so that each field returns a 4 x ny x nx array.
    #    # accumulated fields have been pre-processed s.t. data[:, j, :, :] has accumulation between times j and j+1

    #    all_data_mean = np.nansum(all_data, axis=0)
    #    all_data_sd = np.nanstd(all_data, axis=0)
    #
    #    data1 = all_data_mean  # [:, time_idx, :, :]
    #    data2 = all_data_sd  # [:, time_idx, :, :]
    #    data3 = np.zeros(data1.shape)
    #    data = np.stack([data1, data2, data3, data3], axis=-1)
    #else:
    #    # return mean_start, sd_start, mean_end, sd_end
    #    all_data_mean = np.nanmean(all_data.reshape(2, -1, idx_x, idx_y), axis=1)
    #    all_data_sd = np.nanstd(all_data.reshape(2, -1, idx_x, idx_y), axis=1)

    #    # temp_data_mean = all_data_mean[fcst_idx, time_idx:time_idx+2, :, :]
    #    # temp_data_sd = all_data_sd[fcst_idx, time_idx:time_idx+2, :, :]
    #    data1 = all_data_mean[0, :, :]
    #    data2 = all_data_sd[0, :, :]
    #    data3 = all_data_mean[1, :, :]
    #    data4 = all_data_sd[1, :, :]
    #    data = np.stack([data1, data2, data3, data4], axis=-1)

    data1 = all_data[int((time_idx-30)/3)]
    data2 = all_data[int(1+((time_idx-30)/3))]
    data3 = all_data[int(2+((time_idx-30)/3))]
    data4 = np.nanmean(all_data[int((time_idx-30)/3):int(3+((time_idx-30)/3)),:,:], axis=0)
    data = np.stack([data1, data2, data3, data4], axis=-1)
    
    nc_file.close()

    grid_in = {"lon": lon_reg, "lat": lat_reg, "lon_b": lon_reg_b, "lat_b": lat_reg_b}

    # output grid has a larger coverage and finer resolution
    grid_out = {
        "lon": lon_reg_IMERG,
        "lat": lat_reg_IMERG,
        "lon_b": lon_reg_IMERG_b,
        "lat_b": lat_reg_IMERG_b,
    }

    regridder = xesmf.Regridder(grid_in, grid_out, "conservative")

    if field in nonnegative_fields:
        data = np.maximum(data, 0.0)  # eliminate any data weirdness/regridding issues

    if field in ["Convective precipitation (water)", "Total Precipitation"]:
        #print(field)
        # precip is measured in metres, so multiply to get mm
        #data *= 1000
        #data /= HOURS  # convert to mm/hr
        data = data
    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data /= HOURS * 3600  # convert from a 6-hr difference to a per-second rate

    if (
        field in ["Convective precipitation (water)", "Total Precipitation"]
        and log_precip
    ):
        #print(field)
        data = np.moveaxis(regridder(np.moveaxis(data, -1, 0)), 0, -1)

        return logprec(data, log_precip)

    elif norm:
        # apply transformation to make fields O(1), based on historical
        # forecast data from one of the training years
        if fcst_norm is None:
            raise RuntimeError("Forecast normalisation dictionary has not been loaded")
        if field in ["Medium cloud cover"]:
            # already 0-1

            data = np.moveaxis(regridder(np.moveaxis(data, -1, 0)), 0, -1)

            return data

        elif field in ["Surface pressure", "2 metre temperature"]:
            # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
            data[:, :, :] -= fcst_norm[field]["mean"]
            data /= fcst_norm[field]["std"]
            data = np.moveaxis(regridder(np.moveaxis(data, -1, 0)), 0, -1)

            return data

        elif field in nonnegative_fields:
            data /= fcst_norm[field]["max"]

            data = np.moveaxis(regridder(np.moveaxis(data, -1, 0)), 0, -1)
            # print(data.shape)

            return data

        elif field in ["U component of wind", "V component of wind"]:
            # winds
            data /= max(-fcst_norm[field]["min"], fcst_norm[field]["max"])

            data = np.moveaxis(regridder(np.moveaxis(data, -1, 0)), 0, -1)
            return data

        else:
            data = np.moveaxis(regridder(np.moveaxis(data, -1, 0)), 0, -1)

            return data


def load_fcst_stack(fields, date, time_idx, log_precip=False, norm=False, consolidated=False):
    """
    Returns forecast fields, for the given date and time interval.
    Each field returned by load_fcst has two channels (see load_fcst for details),
    then these are concatentated to form an array of H x W x 4*len(fields)
    """
    field_arrays = []
    for f in fields.keys():
        #print(f)
        field_arrays.append(
            load_fcst(f, date, time_idx, log_precip=log_precip, norm=norm, consolidated=consolidated)
        )
    return np.concatenate(field_arrays, axis=-1)


def get_fcst_stats_slow(field, year=2018):
    """
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done via the data loading routines, which is
    slightly inefficient.
    """
    dates = get_dates(year, start_hour=0, end_hour=168)

    mi = 0.0
    mx = 0.0
    dsum = 0.0
    dsqrsum = 0.0
    nsamples = 0
    for datestr in dates:
        for time_idx in range(28):
            data = load_fcst(field, datestr, time_idx)[:, :, 0]
            mi = min(mi, data.min())
            mx = max(mx, data.max())
            dsum += np.mean(data)
            dsqrsum += np.mean(np.square(data))
            nsamples += 1
    mn = dsum / nsamples
    sd = (dsqrsum / nsamples - mn**2) ** 0.5
    return mi, mx, mn, sd


def get_fcst_stats_fast(field, year=2018):
    """
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done directly from the forecast netcdf file,
    which is somewhat faster, as long as it fits into memory.
    """
    # ds_path = os.path.join(FCST_PATH, str(year), f"{field}.nc")
    # nc_file = nc.Dataset(ds_path, mode="r")

    ds_path = glob.glob(
        FCST_PATH
        + f"gfs{str(year)}*_t00z_f030_f054_{field.replace(' ','-')}_{all_fcst_levels[field]}.zarr"
    )

    z2z = [ZarrToZarr(ds).translate() for ds in ds_path]

    mode_length = np.array([len(z.keys()) for z in z2z]).flatten()
    modals, counts = np.unique(mode_length, return_counts=True)
    index = np.argmax(counts)

    z2zs = [z for z in z2z if len(z.keys()) == modals[index]]

    mzz = MultiZarrToZarr(
        z2zs,
        concat_dims=["time"],
        identical_dims=["step", "latitude", "longitude", all_fcst_levels[field]],
    )

    ref = mzz.translate()

    backend_kwargs = {
        "consolidated": False,
        "storage_options": {
            "fo": ref,
        },
    }
    nc_file = xr.open_dataset(
        "reference://", engine="zarr", backend_kwargs=backend_kwargs
    ).sel({"latitude": lat_reg, "longitude": lon_reg})

    short_names = list(nc_file.variables.keys())

    short_name = [
        short_name
        for short_name in short_names
        if short_name
        not in [
            "latitude",
            "longitude",
            "step",
            all_fcst_levels[field],
            "surface",
            "time",
            "valid_time",
        ]
    ][0]

    if all_fcst_levels[field] == "isobaricInhPa":
        nc_file.sel({"isobaricInhPa": 200})

    # if field in accumulated_fields:
    #    data = nc_file[f"{field}_mean"][:, :-1, :, :]  # last time_idx is full of zeros
    # else:
    #    data = nc_file[f"{field}_mean"][:, :, :, :]

    nc_file.close()

    if field in ["Convective precipitation (water)", "Total precipitation"]:
        # precip is measured in metres, so multiply to get mm
        data = (
            nc_file[short_name]
            .sel({"step": nc_file.step.values[:-1]})
            .sum("step")
            .values
        )
        data *= 1000
        data /= HOURS  # convert to mm/hr
        data = np.maximum(data, 0.0)  # shouldn't be necessary, but just in case

    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data /= HOURS * 3600  # convert from a 6-hr difference to a per-second rate

    else:
        data = nc_file[short_name].values[:, :-1, :, :]

        idx_x = data.shape[2]
        idx_y = data.shape[3]

        data = data.reshape(data.shape[0], 2, -1, idx_x, idx_y)
        data = np.mean(data, axis=2)

    mi = data.min()
    mx = data.max()
    mn = np.mean(data, dtype=np.float64)
    sd = np.std(data, dtype=np.float64)
    return mi, mx, mn, sd


def gen_fcst_norm(year=2018):
    """
    One-off function, used to generate normalisation constants, which
    are used to normalise the various input fields for training/inference.
    """

    stats_dic = {}
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")

    # make sure we can actually write there, before doing computation!!!
    with open(fcstnorm_path, "wb") as f:
        pickle.dump(stats_dic, f)

    start_time = time.time()
    for field in all_fcst_fields:
        print(field)
        mi, mx, mn, sd = get_fcst_stats_fast(field, year)
        stats_dic[field] = {}
        stats_dic[field]["min"] = mi
        stats_dic[field]["max"] = mx
        stats_dic[field]["mean"] = mn
        stats_dic[field]["std"] = sd
        print(
            "Got normalisation constants for",
            field,
            " in ----",
            time.time() - start_time,
            "s----",
        )

    with open(fcstnorm_path, "wb") as f:
        pickle.dump(stats_dic, f)


def load_fcst_norm(year=2018):
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, "rb") as f:
        return pickle.load(f)


try:
    fcst_norm = load_fcst_norm(2021)
    # print(fcst_norm.keys())
except:  # noqa
    fcst_norm = None
