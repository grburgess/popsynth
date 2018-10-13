import h5py
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
importlib.import_module('mpl_toolkits.mplot3d').Axes3D
import pandas as pd
from IPython.display import display, Math, Markdown

from popsynth.utils.spherical_geometry import sample_theta_phi, xyz

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
                 r_max,
                 boundary,
                 strength,
                 n_model,
                 lf_params,
                 spatial_params=None,
                 model_spaces=None,
                 seed=1234,
                 name=None,
                 spatial_form=None,
                 lf_form=None,
                 auxiliary_quantities=None
    ):

        self._luminosities = luminosities
        self._distances = distances
        self._fluxes = fluxes
        self._flux_obs = flux_obs
        self._selection = selection
        self._r_max = r_max
        self._flux_sigma = flux_sigma
        self._boundary = boundary
        self._strength = strength
        self._seed = seed
        self._n_model = n_model
        self._name = name
        self._spatial_form = spatial_form
        self._lf_form = lf_form
        
        self._flux_selected = flux_obs[selection]
        self._distance_selected = distances[selection]
        self._luminosity_selected = luminosities[selection]

        self._flux_hidden = flux_obs[~selection]
        self._distance_hidden = distances[~selection]
        self._luminosity_hidden = luminosities[~selection]

        
        self._lf_params = lf_params
        self._spatial_params = spatial_params

        self._model_spaces = model_spaces


        
        
        if auxiliary_quantities is not None:

            for k,v in auxiliary_quantities.items():

                setattr(self, k, v['true_values'])
                setattr(self,'%s_obs'%k, v['obs_values'])
                setattr(self,'%s_selected'%k, v['obs_values'][selection])


        self._auxiliary_quantites = auxiliary_quantities

                
            
            
        
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
        """
        Create Stan input
        """

        # create a dict for Stan
        output = dict(
            N = self._selection.sum(),
            z_obs = self._distance_selected,
            log_flux_obs = np.log10(self._flux_selected),
            flux_sigma = self._flux_sigma,
            z_max = self._r_max,
            N_model = self._n_model,
            boundary = self._boundary,
            strength = self._strength
        )

        # now append the model spaces
        for k, v in self._model_spaces.items():

            output[k] = v

        for k, v in self._auxiliary_quantites.items():

            output['%s_obs'%k] = v['obs_values'][self._selection]
            output['%s_sigma'%k] = v['sigma']

            
        return output
        

        

    def writeto(self, file_name):

        with h5py.File(file_name, 'w') as f:

            spatial_grp = f.create_group('spatial_params')

            for k,v in self._spatial_params.items():

                spatial_grp.create_dataset(k, data=np.array([v]), compression='lzf')

            lum_grp = f.create_group('lf_params')

            for k,v in self._lf_params.items():

                lum_grp.create_dataset(k, data=np.array([v]), compression='lzf')
            

            f.attrs['name'] = np.string_(self._name)
            f.attrs['lf_form'] = np.string_(self._lf_form)
            f.attrs['spatial_form'] = np.string_(self._spatial_form)
            f.attrs['flux_sigma'] = self._flux_sigma
            f.attrs['n_model'] = self._n_model
            f.attrs['r_max'] = self._r_max
            f.attrs['boundary'] = self._boundary
            f.attrs['strength'] = self._strength
            f.attrs['seed'] = int(self._seed)

            f.create_dataset('luminosities', data=self._luminosities, compression='lzf')
            f.create_dataset('distances', data=self._distances, compression='lzf')
            f.create_dataset('fluxes', data=self._fluxes, compression='lzf')
            f.create_dataset('flux_obs', data=self._flux_obs, compression='lzf')
            f.create_dataset('selection', data=self._selection, compression='lzf')

            aux_grp = f.create_group('auxiliary_quantities')

            for k, v in self._auxiliary_quantites.items():

                q_grp = aux_grp.create_group(k)
                q_grp.create_dataset('true_values',data=v['true_values'], compression='lzf')
                q_grp.create_dataset('obs_values',data=v['obs_values'], compression='lzf')

                q_grp.attrs['sigma'] = v['sigma']
            
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
            r_max = f.attrs['r_max']
            seed = int(f.attrs['seed'])
            name = f.attrs['name']
            lf_form = str(f.attrs['lf_form'])
            spatial_form = str(f.attrs['spatial_form'])
            
            luminosities = f['luminosities'].value
            distances = f['distances'].value
            fluxes = f['fluxes'].value
            flux_obs = f['flux_obs'].value
            selection = f['selection'].value

            model_spaces = {}

            for k in f['model_spaces'].keys():

                model_spaces[str(k)] = f['model_spaces'][k].value

            auxiliary_quantities = {}

            for k in f['auxiliary_quantities'].keys():

                auxiliary_quantities[str(k)] = {'true_values': f['auxiliary_quantities'][k]['true_values'].value,
                                           'obs_values': f['auxiliary_quantities'][k]['obs_values'].value,
                                           'sigma': f['auxiliary_quantities'][k].attrs['sigma']

                }

                
        return cls(
            luminosities=luminosities,
            distances=distances,
            fluxes=fluxes,
            flux_obs=flux_obs,
            selection=selection,
            flux_sigma=flux_sigma,
            n_model=n_model,
            r_max=r_max,
            boundary=boundary,
            strength=strength,
            lf_params=lf_params,
            spatial_params=spatial_params,
            model_spaces=model_spaces,
            seed=seed,
            name=name,
            spatial_form=spatial_form,
            lf_form=lf_form,
            auxiliary_quantities=auxiliary_quantities
            
        )

    def display(self):

        """
        Display the simulation parameters
        
        """


        info = '### %s simulation\nDetected %d out of %d objects' %(self._name, sum(self._selection), len(self._fluxes))

        display(Markdown(info))
        
        out={'parameter':[], 'value':[]}

        display(Markdown('## Luminosity Function'))
        for k,v in self._lf_params.items():

             out['parameter'].append(k)
             out['value'].append(v)

        display(Math(self._lf_form))
        display(pd.DataFrame(out))
        out={'parameter':[], 'value':[]}

        display(Markdown('## Spatial Function'))


        for k,v in self._spatial_params.items():

            out['parameter'].append(k)
            out['value'].append(v)

        display(Math(self._spatial_form))
        display(pd.DataFrame(out))


    def display_true_fluxes(self, ax=None, flux_color = orange):

        if ax is None:
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        ax.scatter(self._distances, self._fluxes, alpha=.5,color=flux_color, edgecolors='none',s=10)

        ax.axhline(self._boundary, color='grey',zorder=-5000,ls='--')
        

        #ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylim(bottom=min([self._fluxes.min(), self._flux_selected.min()]))


        ax.set_xlabel('distance')
        ax.set_ylabel('flux')
        
    def display_obs_fluxes(self, ax=None, flux_color = green):

        if ax is None:
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()


        ax.scatter(self._distance_selected, self._flux_selected , alpha=.8,color=flux_color, edgecolors='none', s=15)


        ax.axhline(self._boundary, color='grey',zorder=-5000,ls='--')
        #ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylim(bottom=min([self._fluxes.min(), self._flux_selected.min()]))
        ax.set_xlim(right=self._r_max)

        ax.set_xlabel('distance')
        ax.set_ylabel('flux')



    def display_fluxes(self, ax=None, true_color=orange, obs_color=green):

        if ax is None:
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        self.display_true_fluxes(ax=ax,flux_color=true_color)
        self.display_obs_fluxes(ax=ax,flux_color=obs_color)
        
        for start, stop, z in zip(self._fluxes[self._selection], self._flux_selected, self._distance_selected):

            x=z
            y=start
            dx=0
            dy = stop-start

            ax.arrow(x, y, dx, dy,color='k', head_width=0.05, head_length=0.2*np.abs(dy),length_includes_head=True )

    def display_obs_fluxes_sphere(self, ax=None, cmap='magma', distance_transform = None, use_log=False):

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        else:

            fig = ax.get_figure()


        n = sum(self._selection)

        theta, phi = sample_theta_phi(n)


        if distance_transform is not None:

            distance = distance_transform(self._distance_selected)

        else:

            distance = self._distance_selected

            
        
        x, y, z = xyz(distance, theta, phi)

        R = self._r_max

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x2 = R * np.outer(np.cos(u), np.sin(v))
        y2 = R * np.outer(np.sin(u), np.sin(v))
        z2 = R * np.outer(np.ones(np.size(u)), np.cos(v))

        
        if use_log:

            x=np.log10(x)
            y=np.log10(y)
            z=np.log10(z)

            x2=np.log10(x2)
            y2=np.log10(y2)
            z2=np.log10(z2)

            R=np.log10(R)
            
        ax.scatter3D(x,y,z,c=self._flux_selected, cmap=cmap,norm=mpl.colors.LogNorm(vmin=min(self._flux_selected), 
                                                                                  vmax=max(self._flux_selected)))

        ax.plot_wireframe(x2, y2, z2, color='grey', alpha=0.9, rcount=4, ccount=2)

        ax._axis3don = False
        ax.set_xlim(-R,R)
        ax.set_ylim(-R,R)
        ax.set_zlim(-R,R)


    def display_fluxes_sphere(self, ax=None, cmap='magma', distance_transform = None, use_log=False):

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        else:

            fig = ax.get_figure()


        n = len(self._fluxes)

        theta, phi = sample_theta_phi(n)


        if distance_transform is not None:

            distance = distance_transform(self._distances)

        else:

            distance = self._distances

            
        
        x, y, z = xyz(distance, theta, phi)

        R = self._r_max

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x2 = R * np.outer(np.cos(u), np.sin(v))
        y2 = R * np.outer(np.sin(u), np.sin(v))
        z2 = R * np.outer(np.ones(np.size(u)), np.cos(v))

        
        if use_log:

            x=np.log10(x)
            y=np.log10(y)
            z=np.log10(z)

            x2=np.log10(x2)
            y2=np.log10(y2)
            z2=np.log10(z2)

            R=np.log10(R)
            
            
        ax.scatter3D(x,y,z,c=self._flux_obs, cmap=cmap,norm=mpl.colors.LogNorm(vmin=min(self._fluxes), 
                                                                                  vmax=max(self._fluxes)))
        
        ax.plot_wireframe(x2, y2, z2, color='grey', alpha=0.9, rcount=4, ccount=2)

        ax._axis3don = False
        ax.set_xlim(-R,R)
        ax.set_ylim(-R,R)
        ax.set_zlim(-R,R)

    def display_hidden_fluxes_sphere(self, ax=None, cmap='magma', distance_transform = None, use_log=False):

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        else:

            fig = ax.get_figure()


        n = len(self._flux_hidden)

        theta, phi = sample_theta_phi(n)


        if distance_transform is not None:

            distance = distance_transform(self._distance_hidden)

        else:

            distance = self._distance_hidden

            
        
        x, y, z = xyz(distance, theta, phi)

        R = self._r_max

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x2 = R * np.outer(np.cos(u), np.sin(v))
        y2 = R * np.outer(np.sin(u), np.sin(v))
        z2 = R * np.outer(np.ones(np.size(u)), np.cos(v))

        
        if use_log:

            x=np.log10(x)
            y=np.log10(y)
            z=np.log10(z)

            x2=np.log10(x2)
            y2=np.log10(y2)
            z2=np.log10(z2)
            R=np.log10(R)
            
        ax.scatter3D(x,y,z,c=self._flux_hidden, cmap=cmap,norm=mpl.colors.LogNorm(vmin=min(self._flux_hidden), 
                                                                                  vmax=max(self._flux_hidden)))
        
        ax.plot_wireframe(x2, y2, z2, color='grey', alpha=0.9, rcount=4, ccount=2)

        ax._axis3don = False
        ax.set_xlim(-R,R)
        ax.set_ylim(-R,R)
        ax.set_zlim(-R,R)

    def display_flux_sphere(self, ax=None, seen_cmap='magma', unseen_cmap='Greys' , distance_transform = None, use_log=False):

        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        else:
            fig = ax.get_figure()


        self.display_obs_fluxes_sphere(ax=ax,cmap=seen_cmap,distance_transform=distance_transform,use_log=use_log)
        self.display_hidden_fluxes_sphere(ax=ax,cmap=unseen_cmap,distance_transform=distance_transform,use_log=use_log)

    def display_luminosty(self):

        fig, ax = plt.subplots()
        
        bins = np.logspace(np.log10(self._luminosities.min()),np.log10(self._luminosities.max()),30)

        ax.hist(self._luminosities,bins=bins,normed=True,facecolor=orange,edgecolor=orange_highlight,lw=1.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('L')


    def display_distances(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        else:
            fig = ax.get_figure()

        bins = np.linspace(0,self._r_max,40)
        ax.hist(data['z'],bins=bins, facecolor=green, edgecolor=green_highlight, lw=1.5,label='Total Pop.')
        ax.hist(data['z_obs'],bins=bins, facecolor=blue, edgecolor=blue_highlight, lw=1.5,alpha=1,label='Obs. Pop.')

        ax.set_xlabel('z')
        ax.legend()
        #sns.despine(offset=5, trim=True);
