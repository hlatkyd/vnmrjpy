import vnmrjpy as vj
import os
import json
import matplotlib.pyplot as plt
import numpy as np

"""
Collection of unsorted utility functions and classes. Includes:
    
Functions:
    vprint -- print if verbose is True
    savepd -- save procpar dictionary to json
    loadpd -- load procpar dictionary from json

Classes:
    FitViewer3D -- view 4D volume along with best fit on time axis
                and parameter maps. Similar to the nibabel viewer
                but slower and less useful all-around
"""

def vprint(string):

    if vj.config['verbose']==True:
        print(string)
    else:
        pass


def getpetab(pd):

    pass 

def savepd(pd,out):
    """Save procpar dictionary to json file"""
    # enforce .json 
    if out.endswith('.json'):
        pass
    else:
        out = out+'.json'
    # check basedir:
    basedir = out.rsplit('/',1)[0]
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(out,'w') as openfile:
        jsondata = json.dumps(pd)
        json.dump(pd,openfile)

def loadpd(infile):
    """Load procpar dictionary from json file"""
    with open(infile, 'r') as openfile:
        pd = json.load(openfile)
    return pd

class FitViewer3D():
    """Draw a 3D volume along with regression lines and parameter maps

    Args:
        image -- 4D or numpy array
        t_axis -- time points in 1D array (the x axis for the 'model')
        model -- python callable function of form (x, param1, param2 etc)
        fit_params -- list of 3D numpy arrays of fit parameters 

    Methods:
        plot()
    
    """

    def __init__(self, image, t_axis, model, fit_params):

        if image.ndim > 4:
            raise Exception
        self.image = image
        self.model = model
        self.params = fit_params
        self.t_axis = t_axis
        self.res = 100  # fitting grid resolution

    def plot(self, plot_params=True):
        
        def _draw_images():
            
            (ix,iy,iz,it) = self.coords
            for i in range(4):
                ax[0,i].clear()
            ax[0,0].imshow(self.image[:,:,iz,it], cmap='gray')
            ax[0,1].imshow(self.image[:,iy,:,it], cmap='gray')
            ax[0,2].imshow(self.image[ix,:,:,it], cmap='gray')

        def _draw_param_slices():

            (ix,iy,iz,it) = self.coords
            for i in range(len(self.params)):
                ax[1,i].clear()
            for k in range(len(self.params)):
                ax[1,k].imshow(self.params[k][:,:,iz], cmap='hot')
        #TODO
        def _update_images():
            """Just update data on imshows"""
            pass

        def _draw_crosshairs():
            try:
                for crossh in self.crosshairs:
                    crossh.remove()
            except: pass
            chcolor = 'green'
            (ix,iy,iz,it) = self.coords
            vline_01 = ax[0,0].axvline(x=ix,color=chcolor)
            hline_01 = ax[0,0].axhline(y=iy,color=chcolor)
            vline_02 = ax[0,1].axvline(x=ix,color=chcolor)
            hline_02 = ax[0,1].axhline(y=iz,color=chcolor)
            vline_03 = ax[0,2].axvline(x=iy,color=chcolor)
            hline_03 = ax[0,2].axhline(y=iz,color=chcolor)
            vline_04 = ax[0,3].axvline(x=it, color=chcolor)
            for i in [vline_01,hline_01,vline_02,hline_02,vline_03,\
                                hline_03,vline_04]:
                self.crosshairs.append(i)

        def _draw_best_fit():

            (ix,iy,iz,it) = self.coords
            t_grid = np.linspace(self.t_axis[0],self.t_axis[-1],self.res)
            params = [i[ix,iy,iz] for i in self.params]
            best_fit = self.model(t_grid, *params)

            # also raw points
            ax[0,3].plot(self.t_axis, self.image[ix,iy,iz,:],'bo')
            ax[0,3].plot(t_grid,best_fit,'r-')
            ax[0,3].set_aspect('equal')

        def _on_click(event):

            if event.inaxes == ax[0,0]:
                self.coords[0] = event.xdata
                self.coords[1] = event.ydata
            elif event.inaxes == ax[0,1]:
                self.coords[0] = event.xdata
                self.coords[2] = event.ydata
            elif event.inaxes == ax[0,2]: 
                self.coords[1] = event.xdata
                self.coords[2] = event.ydata
            elif event.inaxes == ax[0,3]: 
                self.coords[3] = event.xdata
            else:
                pass
            # reading new coordinates
            self.coords = [int(i) for i in self.coords]
            _draw_images()
            _draw_param_slices()
            _draw_crosshairs()
            _draw_best_fit()
            plt.draw()

        # init coordinates
        self.coords = [0,0,0,0]
        self.crosshairs = []

        param_num = len(self.params)
        row_num = param_num // 4 + 2
            
        # volume slice plots
        fig, ax = plt.subplots(nrows=row_num, ncols=4, figsize=(10,5))
        # hide unused subplots
        for i in range(param_num, 4):
            ax[1,i].axis('off')
        # draw with initial coords 
        _draw_images()
        _draw_crosshairs()
        _draw_best_fit()

        fig.canvas.mpl_connect('button_press_event', _on_click)
        plt.show()
