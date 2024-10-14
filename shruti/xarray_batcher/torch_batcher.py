import torch
import xbatcher
import dask
import numpy as np
from tqdm.dask import TqdmCallback
from tqdm import tqdm
from scipy.spatial import KDTree

def get_spherical(lat,lon, elev):

    """
    Get spherical coordinates of lat and lon, not assuming unit ball for radius
    So we also take elev into account

    Inputs
    ------

    lat: np.array or xr.DataArray (n_lats,n_lons)
         meshgrid of latitude points

    lon: np.array or xr.DataArray (n_lats,n_lons)
         meshgrid of longitude points

    elev: np.array or xr.DataArray (n_lats,n_lons)
          altitude values in m

    Output
    ------

    r, sigma and phi
    See: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    for more details
    """
    
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    x = elev * np.cos(lat) * np.cos(lon)
    y = elev * np.cos(lat) * np.sin(lon)
    z = elev * np.sin(lat)
    
    return np.hstack((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)))

class BatchDataset(torch.utils.data.Dataset):

    """
    class for iterating over a dataset
    """
    def __init__(self, X, y, constants, batch_size=[4, 128, 128], weighted_sampler=True, for_NJ=False, for_val=False):

        self.batch_size=batch_size
        self.X_generator = X
        self.y_generator = xbatcher.BatchGenerator(y,
                {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[1]},
                input_overlap={"lat": int(batch_size[1]/2), "lon": int(batch_size[1]/2)})
        constants["lat"]=np.round(y.lat.values,decimals=2)
        constants["lon"]=np.round(y.lon.values,decimals=2)
                                                  
        self.constants_generator = constants
        
        self.variables = [list(x.data_vars)[0] for x in X]
        self.constants = list(constants.data_vars)
        self.for_NJ = for_NJ
        self.for_val = for_val

        if weighted_sampler:
            y_train = [self.y_generator[i].precipitation.mean(["time","lat","lon"], skipna=False)\
                       for i in range(len(self.y_generator))]
            class_sample_count = np.array(
                                [len(np.where(np.round(y_train,decimals=1) == t)[0])\
                                 for t in np.unique(np.round(y_train,decimals=1))
                                ]
            )
            weight = 1. / class_sample_count
            samples_weight = np.zeros_like(y_train)
            for i_t,t in \
            enumerate(\
                np.unique(np.round(y_train,decimals=1))):
                idx = np.squeeze(np.argwhere(np.round(y_train,decimals=1) == t))
                samples_weight[idx]=weight[i_t]
                
            self.samples_weight = torch.from_numpy(np.asarray(samples_weight))
            self.sampler = torch.utils.data.WeightedRandomSampler(\
                self.samples_weight.type('torch.DoubleTensor'), 
                len(samples_weight)
            )
                
    def __len__(self) -> int:
        return len(self.y_generator)

    def __getitem__(self, idx):

        y_batch = self.y_generator[idx]
        time_batch = y_batch.time.values
        lat_batch = np.round(y_batch.lat.values, decimals=2)
        lon_batch = np.round(y_batch.lon.values, decimals=2)

        X_batch = []
        for x,variable in zip(self.X_generator,self.variables):
            try:
                X_batch.append(x[variable].sel({"time":time_batch,
                                          "lat":lat_batch,
                                         "lon":lon_batch}).values)
            except:
                print(variable,time_batch)
                X_batch.append(np.zeros([len(time_batch),len(lat_batch),len(lon_batch),4]))    
            

        X_batch = torch.tensor(
            np.concatenate(X_batch, axis=-1,
                ), dtype=torch.float32)

        constant_batch = torch.tensor(
            np.stack( 
                        [
                            self.constants_generator[constant].sel({"time":0,"lat":lat_batch,
                                     "lon":lon_batch}).values for constant in self.constants], axis=-1,
                    ), dtype=torch.float32
        )


        if self.for_NJ:
            
            elev_values = np.squeeze(constant_batch[:,:,0]).reshape(-1,1)
            lat_values, lon_values = np.meshgrid(lat_batch, lon_batch)
            spherical_coords = get_spherical(lat_values.reshape(-1,1),lon_values.reshape(-1,1),elev_values)
            
            kdtree = KDTree(spherical_coords)

            pairs = []

            for i_coord, coord in enumerate(spherical_coords):
                pairs.append(np.vstack((np.full(3,fill_value=i_coord).reshape(1,-1),kdtree.query(coord, k=3)[1])))

            pairs = np.hstack((pairs))
                
            rainfall_path = torch.cat((torch.tensor(\
                y_batch.precipitation.fillna(0).values.reshape(self.batch_size[0],-1,1), dtype=torch.float32
                ),X_batch.reshape(self.batch_size[0],-1,len(self.variables)*4))
                                      ,dim=-1)
            obs_dates = np.ones(self.batch_size[0]).reshape(1,-1)
            n_obs = np.array([self.batch_size[0]])
            if self.for_val:
                obs_dates = np.zeros(self.batch_size[0]).reshape(1,-1)
                n_obs = np.random.randint(1,self.batch_size[0]-8,1)
                obs_dates[:n_obs[0]]=1
            
            return{"idx": idx, "rainfall_path": rainfall_path[None,:,:,:],
                "observed_dates": obs_dates, 
                "nb_obs": n_obs, "dt": 1, "edge_indices": pairs,
                "obs_noise": None}

        else:
            return (torch.cat((X_batch,constant_batch),dim=-1), torch.tensor(
                y_batch.precipitation.fillna(0).values[:,:,:,None], dtype=torch.float32
            ))

class BatchTruth(torch.utils.data.Dataset):

    """
    class for iterating over a dataset
    """
    def __init__(self, y, batch_size=[4, 128, 128], weighted_sampler=True):

        self.batch_size=batch_size
        self.y_generator = xbatcher.BatchGenerator(y,
                {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[1]},
                input_overlap={"lat": int(batch_size[1]/2), "lon": int(batch_size[1]/2)})

        if weighted_sampler:
            y_train = [self.y_generator[i].precipitation.mean(["time","lat","lon"], skipna=False)\
                       for i in range(len(self.y_generator))]
            class_sample_count = np.array(
                                [len(np.where(np.round(y_train,decimals=1) == t)[0])\
                                 for t in np.unique(np.round(y_train,decimals=1))
                                ]
            )
            weight = 1. / class_sample_count
            samples_weight = np.zeros_like(y_train)
            for i_t,t in \
            enumerate(\
                np.unique(np.round(y_train,decimals=1))):
                idx = np.squeeze(np.argwhere(np.round(y_train,decimals=1) == t))
                samples_weight[idx]=weight[i_t]
                
            self.samples_weight = torch.from_numpy(np.asarray(samples_weight))
            self.sampler = torch.utils.data.WeightedRandomSampler(\
                self.samples_weight.type('torch.DoubleTensor'), 
                len(samples_weight)
            )
                
    def __len__(self) -> int:
        return len(self.y_generator)

    def __getitem__(self, idx):

        y_batch = self.y_generator[idx]

        return (torch.tensor(
                y_batch.precipitation.fillna(0).values[:,:,:,None], dtype=torch.float32
            ))

        
        

        
        

        

