# Big warning:
# This is not a general-purpose forecast script.
# This is for forecasting on the pre-defined 'ICPAC region' (e.g., the latitudes
# and longitudes are hard-coded), and assumes the input forecast data starts at
# time 0, with time steps of data.HOURS.
# A more robust version of this script would parse the latitudes, longitudes, and
# forecast time info from the input file.
# The forecast data fields must match those defined in data.all_fcst_fields
import os
import pathlib
import yaml

import netCDF4 as nc
import xarray as xr
import numpy as np
from tensorflow.keras.utils import Progbar

from data import (
    HOURS,
    all_fcst_fields,
    accumulated_fields,
    nonnegative_fields,
    fcst_norm,
    logprec,
    denormalise,
    load_hires_constants,
)
import read_config
from noise import NoiseGenerator
from setupmodel import setup_model

from datetime import datetime, timedelta

# %%
# Define the latitude and longitude arrays for later
latitude = np.arange(-13.65, 24.7, 0.1)
longitude = np.arange(19.15, 54.3, 0.1)

# Some setup
read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
data_paths = read_config.get_data_paths()  # need the constants directory
downscaling_steps = read_config.read_downscaling_factor()["steps"]
assert fcst_norm is not None

# %%
# Open and parse forecast.yaml
#fcstyaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast.yaml")
fcstyaml_path = "forecast_gfs.yaml"
with open(fcstyaml_path, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# %%
model_folder = fcst_params["MODEL"]["folder"]
checkpoint = fcst_params["MODEL"]["checkpoint"]
input_folder = fcst_params["INPUT"]["folder"]
#input_folder = "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IFS-regICPAC-meansd/2023/"
input_files = fcst_params["INPUT"]["file"]
start_hour = fcst_params["INPUT"]["start_hour"]
end_hour = fcst_params["INPUT"]["end_hour"]
output_folder = fcst_params["OUTPUT"]["folder"]
ensemble_members = fcst_params["OUTPUT"]["ensemble_members"]

assert start_hour % HOURS == 0, f"start_hour must be divisible by {HOURS}"
assert end_hour % HOURS == 0, f"end_hour must be divisible by {HOURS}"

# Open and parse GAN config file
config_path = os.path.join(model_folder, "setup_params.yaml")
with open(config_path, "r") as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]  # TODO: avoid setting up discriminator in forecast mode?
constant_fields = 2

assert mode == "GAN", "standalone forecast script only for GAN, not VAE-GAN or deterministic model"

# Set up pre-trained GAN
weights_fn = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07}.h5")
input_channels = 4*len(all_fcst_fields)

model = setup_model(mode=mode,
                    arch=arch,
                    downscaling_steps=downscaling_steps,
                    input_channels=input_channels,
                    constant_fields=constant_fields,
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding)
gen = model.gen
gen.load_weights(weights_fn)

network_const_input = load_hires_constants(batch_size=1)  # 1 x lats x lons x 2

# %%
def create_output_file(nc_out_path):
    netcdf_dict = {}
    rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")
    netcdf_dict["rootgrp"] = rootgrp
    rootgrp.description = "GAN 6-hour rainfall ensemble members in the ICPAC region."

    # Create output file dimensions
    rootgrp.createDimension("latitude", len(latitude))
    rootgrp.createDimension("longitude", len(longitude))
    rootgrp.createDimension("member", ensemble_members)
    rootgrp.createDimension("time", None)
    rootgrp.createDimension("valid_time", None)

    # Create variables
    latitude_data = rootgrp.createVariable("latitude",
                                           "f4",
                                           ("latitude",))
    latitude_data.units = "degrees_north"
    latitude_data[:] = latitude     # Write the latitude data

    longitude_data = rootgrp.createVariable("longitude",
                                            "f4",
                                            ("longitude",))
    longitude_data.units = "degrees_east"
    longitude_data[:] = longitude   # Write the longitude data

    ensemble_data = rootgrp.createVariable("member",
                                           "i4",
                                           ("member",))
    ensemble_data.units = "ensemble member"
    ensemble_data[:] = range(1, ensemble_members+1)

    netcdf_dict["time_data"] = rootgrp.createVariable("time",
                                                      "f4",
                                                      ("time",))
    netcdf_dict["time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["valid_time_data"] = rootgrp.createVariable("fcst_valid_time",
                                                            "f4",
                                                            ("time", "valid_time"))
    netcdf_dict["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["precipitation"] = rootgrp.createVariable("precipitation",
                                                          "f4",
                                                          ("time", "member", "valid_time",
                                                           "latitude", "longitude"),
                                                          compression="zlib",
                                                          chunksizes=(1, 1, 1, len(latitude), len(longitude)))
    netcdf_dict["precipitation"].units = "mm h**-1"
    netcdf_dict["precipitation"].long_name = "Precipitation"

    return netcdf_dict

# %%
#input_file
for year in [2023]:
    # %%
    # Open input netCDF file to get the times
    #nc_in_path = os.path.join(input_folder, "cape.nc")
    #print(f"IFS: {nc_in_path}")
    #nc_in = nc.Dataset(nc_in_path, mode="r")
#    start_times = [nc_in["time"][0]]
    #valid_times = nc_in["valid_time"][:]
    valid_times = np.arange(start_hour,end_hour+1,HOURS)
    if year==2023:
        dates = np.arange("%i-06-03"%year,"%i-08-01"%(year),np.timedelta64(1,"D"),dtype="datetime64[D]")

    if year==2024:
        dates = np.arange("%i-08-25"%year,"%i-09-01"%year,np.timedelta64(1,"D"),dtype="datetime64[D]")

    for day in dates:
        d = datetime(int(str(day).split("-")[0]),
                     int(str(day).split("-")[1]),
                     int(str(day).split("-")[2])
                    )#+ timedelta(hours=int(start_times[0]))
        print(f"{d.year}-{d.month:02}-{d.day:02}")
        
        # %%
        # Specify input folder for year
        input_folder_year = input_folder+f"{d.year}/"
        
        # Create output netCDF file
        output_folder_year = output_folder+f"{d.year}/"
        pathlib.Path(output_folder_year).mkdir(parents=True, exist_ok=True)
        nc_out_path = os.path.join(output_folder_year, f"GAN_{d.year}{d.month:02}{d.day:02}.nc")
        
        netcdf_dict = create_output_file(nc_out_path)
        netcdf_dict["time_data"][0] = np.array([f"{d.year}-{d.month:02}-{d.day:02}"],dtype='datetime64[ns]')#start_times[0]
        
        # loop over time chunks. output forecasts may not start from hour 0, so
        # generate output and input valid time indices using enumerate(...)
        for out_time_idx, in_time_idx in enumerate(range(start_hour//HOURS, end_hour//HOURS)):
            # copy across valid_time from input file
            netcdf_dict["valid_time_data"][0, out_time_idx] = valid_times[out_time_idx]
            d_valid = d+ timedelta(hours=int(in_time_idx*HOURS))

            field_arrays = []
        
            # the contents of the next loop are v. similar to load_fcst from data.py,
            # but not quite the same, since that has different assumptions on how the
            # forecast data is stored.  TODO: unify the data normalisation between these?
            for field in all_fcst_fields.keys():
                # Original:
                # nc_in[field] has shape 1 x 50 x 29 x 384 x 352
                # corresponding to n_forecasts x n_ensemble_members x n_valid_times x n_lats x n_lons
                # Ensemble mean:
                # nc_in[field] has shape len(nc_in["time"]) x 29 x 384 x 352
                
                # Open input netCDF file
                input_file = f"{all_fcst_fields[field]}.zarr"
                #input_file = 'IFS_20180606_00Z.nc'
                #input_file = f'IFS_{d.year}{d.month:02}{d.day:02}_00Z.nc'
                nc_in_path = os.path.join(input_folder_year, input_file)
                nc_in = xr.open_zarr(nc_in_path)#.sel({'time':f"{d.year}-{d.month:02}-{d.day:02}"})#nc.Dataset(nc_in_path, mode="r")

                #print(nc_in.time.values)
                data = np.moveaxis(np.squeeze(nc_in.sel({"time":[d_valid]}).to_dataarray().values),0,-1)
                
                field_arrays.append(data)
 
            network_fcst_input = np.concatenate(field_arrays, axis=-1)  # lat x lon x 4*len(all_fcst_fields)
            network_fcst_input = np.expand_dims(network_fcst_input, axis=0)  # 1 x lat x lon x 4*len(...)
        
            noise_shape = network_fcst_input.shape[1:-1] + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=1)
            progbar = Progbar(ensemble_members)
            for ii in range(ensemble_members):
                gan_inputs = [network_fcst_input, network_const_input, noise_gen()]
                gan_prediction = gen.predict(gan_inputs, verbose=False)  # 1 x lat x lon x 1
                netcdf_dict["precipitation"][0, ii, out_time_idx, :, :] = denormalise(gan_prediction[0, :, :, 0])
                progbar.add(1)
        
        netcdf_dict["rootgrp"].close()