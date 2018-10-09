import h5py

class Population(object):

    def __init__(self,luminosities,distances,fluxes, flux_obs, selection, flux_sigma, n_model, lf_params, spatial_params=None, model_spaces=None):
        
        self._luminosities = luminosities
        self._distances = distances
        self._fluxes = fluxes
        self._flux_obs = flux_obs
        self._selection = selection
        self._flux_sigma

        self._n_model = n_model


        self._flux_selected = flux_obs[selection]
        self._distance_selected = distance[selection]
        self._luminosity_selected = luminosites[selection]

        self._lf_params = lf_params
        self._spatial_params = spatial_params

        self._model_spaces = model_spaces

        if model_spaces is not None:
            
            for k,v in model_spaces.items():

                assert len(v) == n_model
        
        
    def to_stan_data(self):

        pass


    def writeto(self, file_name):

        
        with h5py.File(file_name,'w') as f:

            f.attrs['spatial_params'] = self._spatial_params
            f.attrs['lf_params'] = self._lf_params
            f.attrs['flux_sigma'] = self._flux_sigma
            f.attrs['n_model'] = self._n_model

            f.create_dataset('luminosities', data=self._luminosities, compression='lzf')
            f.create_dataset('distances', data=self._distances, compression='lzf')
            f.create_dataset('fluxes', data=self._fluxes, compression='lzf')
            f.create_dataset('flux_obs', data=self._flux_obs, compression='lzf')
            f.create_dataset('selection', data=self._selection, compression='lzf')
            

            model_grp = f.create_group('model_spaces'):

            for k,v in self._model_spaces.items():

                model_grp.create_dataset(k, data=v, compression='lzf')

            
        
    
    @classmethod
    def from_file(cls, file_name):

        with h5py.File(file_name,'r') as f:

            spatial_params = f.attrs['spatial_params']
            lf_params = f.attrs['lf_params']
            flux_sigma = f.attrs['flux_sigma']

            luminosities = f['luminosities'].value
            distances = f['distances'].value
            fluxes = f['fluxes'].value
            flux_obs = f['flux_obs'].value
            selection = f['selection'].value


            model_spaces = {}

            for k in f['model_spaces'].keys():

                model_spaces[k] = f['model_spaces'][k].values
            

        return cls(luminosities=luminosities,
                   distances=distances,
                   fluxes=fluxes,
                   flux_obs=flux_obs,
                   selection=selection,
                   flux_sigma=flux_sigma,
                   n_model=n_model,
                   lf_params=lf_params,
                   spatial_params=spatial_params,
                   model_spaces = model_spaces

        )
        

    def display(self):

        pass
        
