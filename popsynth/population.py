import h5py
import numpy as np
import matplotlib.pyplot as plt


green = '#24B756'
green_highlight = '#07A23B'
orange = '#FA7031'
orange_highlight = '#DD4B09'
blue = '#1E9A91'
blue_highlight = '#06887F'

class Population(object):

    def __init__(self,
                 luminosities,
                 distances,
                 fluxes,
                 flux_obs,
                 selection,
                 flux_sigma,
                 boundary,
                 strength,
                 n_model,
                 lf_params,
                 spatial_params=None,
                 model_spaces=None,
                 seed=1234):

        self._luminosities = luminosities
        self._distances = distances
        self._fluxes = fluxes
        self._flux_obs = flux_obs
        self._selection = selection
        self._flux_sigma = flux_sigma
        self._boundary = boundary
        self._strength = strength
        self._seed = seed
        self._n_model = n_model

        self._flux_selected = flux_obs[selection]
        self._distance_selected = distances[selection]
        self._luminosity_selected = luminosities[selection]

        self._lf_params = lf_params
        self._spatial_params = spatial_params

        self._model_spaces = model_spaces

        if model_spaces is not None:

            for k, v in model_spaces.items():

                assert len(v) == n_model


    @property
    def luminosities(self):
        return self._luminosities


    @property
    def fluxes(self):
        return self._fluxes

    @property
    def distances(self):
        return self._distances

    @property
    def selection(self):
        return self._selection

    @property
    def flux_observed(self):
        return self._flux_obs

    @property
    def selected_fluxes(self):
        return self._flux_selected

    @property
    def selected_distances(self):
        return self._distance_selected

    @property
    def luminosity_parameters(self):
        return self._lf_params

    @property
    def spatial_parameters(self):
        return self._spatial_params

    
    
    def to_stan_data(self):

        pass

    def writeto(self, file_name):

        with h5py.File(file_name, 'w') as f:

            spatial_grp = f.create_group('spatial_params')

            for k,v in self._spatial_params.items():

                spatial_grp.create_dataset(k, data=np.array([v]), compression='lzf')

            lum_grp = f.create_group('lf_params')

            for k,v in self._lf_params.items():

                lum_grp.create_dataset(k, data=np.array([v]), compression='lzf')
            

            f.attrs['flux_sigma'] = self._flux_sigma
            f.attrs['n_model'] = self._n_model
            f.attrs['boundary'] = self._boundary
            f.attrs['strength'] = self._strength
            f.attrs['seed'] = int(self._seed)

            f.create_dataset('luminosities', data=self._luminosities, compression='lzf')
            f.create_dataset('distances', data=self._distances, compression='lzf')
            f.create_dataset('fluxes', data=self._fluxes, compression='lzf')
            f.create_dataset('flux_obs', data=self._flux_obs, compression='lzf')
            f.create_dataset('selection', data=self._selection, compression='lzf')

            model_grp = f.create_group('model_spaces')

            for k, v in self._model_spaces.items():

                model_grp.create_dataset(k, data=v, compression='lzf')

    @classmethod
    def from_file(cls, file_name):

        with h5py.File(file_name, 'r') as f:


            spatial_params = {}
            lf_params = {}

            for key in f['spatial_params'].keys():

                spatial_params[key] = f['spatial_params'][key].value[0]

            for key in f['lf_params'].keys():

                lf_params[key] = f['lf_params'][key].value[0]
            
            flux_sigma = f.attrs['flux_sigma']
            boundary = f.attrs['boundary']
            strength = f.attrs['strength']
            n_model = f.attrs['n_model']
            seed = int(f.attrs['seed'])
            
            luminosities = f['luminosities'].value
            distances = f['distances'].value
            fluxes = f['fluxes'].value
            flux_obs = f['flux_obs'].value
            selection = f['selection'].value

            model_spaces = {}

            for k in f['model_spaces'].keys():

                model_spaces[k] = f['model_spaces'][k].values

        return cls(
            luminosities=luminosities,
            distances=distances,
            fluxes=fluxes,
            flux_obs=flux_obs,
            selection=selection,
            flux_sigma=flux_sigma,
            n_model=n_model,
            boundary=boundary,
            strength=strength,
            lf_params=lf_params,
            spatial_params=spatial_params,
            model_spaces=model_spaces,
            seed=seed)

    def display(self):

        pass

    def display_fluxes(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        ax.scatter(self._distances, self._fluxes, alpha=.2,color=orange, edgecolors='none',s=10)
        ax.scatter(self._distance_selected, self._flux_selected , alpha=.8,color=green,edgecolors='none', s=15)

        for start, stop, z in zip(self._fluxes[self._selection], self._flux_selected, self._distance_selected):
    
            x=z
            y=start
            dx=0
            dy = stop-start
   
            ax.arrow(x, y, dx, dy,color='k', head_width=0.05, head_length=0.2*np.abs(dy),length_includes_head=True )

        ax.axhline(self._boundary, color='grey',zorder=-5000,ls='--')
        

        #ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylim(bottom=min([self._fluxes.min(), self._flux_selected.min()]))
