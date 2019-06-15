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
    change_procpar -- modify parameter value in procpar file

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

def change_procpar(path, par, val, newfile=None):
    """Rewrite procpar parameter in file to the desired value

    Args:
        path -- path to .fid direcory od procpar file
        par -- parameter name, can be a list
        val -- desired value, format should be correct, can be a list
        newfile -- save as new file in same directory, if None, procpar is
                    overwritten
    Return:
        None
    """
    # sanity check for list lengths
    if type(par) == list:
        if type(val) != list:
            raise Exception
        if len(par) != len(val):
            raise Exception
    else:
        par = [par]
        val = [val]
    par = [str(i) for i in par]
    if path[-4:] == '.fid':  # path was fid directory
        ppfile = path+'/procpar'
    elif path[-7:] == 'procpar':
        ppfile = path
    # save file permissions
    stat = os.stat(ppfile)
    uid, gid = stat[4], stat[5]

    # reading file
    line_nums = []  # list of line numbers where pars appear
    par_order = []  # order in which the pars appear in file
    val_order = []
    with open(ppfile,'r') as openpp:
        lines = openpp.readlines()
        for num, line in enumerate(lines):
            if line.startswith(tuple(par)):
                # which par; also, does this make sense?
                for i in par:
                    if line.startswith(i):
                        curpar = i
                        curval = val[par.index(i)]
                line_nums.append(num)
                par_order.append(curpar)
                val_order.append(curval)
    # change lines
    for num, line in enumerate(lines):
        if num in line_nums:
            if lines[num+1].startswith('1'):
                # check value type
                ind = line_nums.index(num)
                if type(val_order[ind]) == str:
                    lines[num+1] = '1 "'+val_order[ind]+'"\n'
                else:
                    lines[num+1] = '1 '+str(val_order[ind])+'\n'
            else:
                raise Exception('Unexpected start of line '+str(num+1))

    # writing file
    if newfile == None:
        newppfile = ppfile 
    else:
        newppfile = ppfile.rsplit('/',1)[0]+'/'+newfile
    with open(newppfile,'w') as openpp:
        openpp.writelines(lines)
    # restore permissions
    os.chown(newppfile, uid ,gid)
    

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

    Utility tool used for fieldmap, wasabi, etc debugging.

    Args:
        image -- 4D or numpy array
        t_axis -- time points in 1D array (the x axis for the 'model')
        model -- python callable function of form (x, param1, param2 etc)
        fit_params -- list of 3D numpy arrays of fit parameters 

    Methods:
        plot()
    
    """

    def __init__(self, image, t_axis, model, fit_params, slc_dim=None):

        if image.ndim > 4:
            raise Exception
        self.image = image
        self.model = model
        self.params = fit_params
        self.t_axis = t_axis
        self.res = 100  # fitting grid resolution
        if slc_dim == None:
            min_dim = min(image.shape[:3])
            slc_dim = image.shape.index(min_dim)

        self.slc_dim = slc_dim
        self.aspect = ['auto','auto','auto']            

    def plot(self, plot_params=True):
        
        def _draw_images():
            
            (ix,iy,iz,it) = self.coords
            for i in range(4):
                ax[0,i].clear()
            ax[0,0].imshow(self.image[:,:,iz,it], cmap='gray')
            ax[0,1].imshow(self.image[:,iy,:,it], cmap='gray')
            ax[0,2].imshow(self.image[ix,:,:,it], cmap='gray')
            for i in range(3):
                ax[0,i].set_aspect(self.aspect[i])

        def _draw_param_slices():

            (ix,iy,iz,it) = self.coords
            _vmin, _vmax = (0,5)
            for i in range(len(self.params)):
                ax[1,i].clear()
            for k in range(len(self.params)):
                if self.slc_dim == 0:
                    ax[1,k].imshow(self.params[k][ix,:,:], cmap='hot',\
                        vmin=_vmin, vmax=_vmax)
                elif self.slc_dim == 1:
                    ax[1,k].imshow(self.params[k][:,iy,:], cmap='hot',\
                        vmin=_vmin, vmax=_vmax)
                elif self.slc_dim == 2:
                    ax[1,k].imshow(self.params[k][:,:,iz], cmap='hot',\
                        vmin=_vmin, vmax=_vmax)
                ax[1,k].set_aspect('auto')
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
            vline_01 = ax[0,0].axvline(x=iy,color=chcolor)
            hline_01 = ax[0,0].axhline(y=ix,color=chcolor)
            vline_02 = ax[0,1].axvline(x=iz,color=chcolor)
            hline_02 = ax[0,1].axhline(y=ix,color=chcolor)
            vline_03 = ax[0,2].axvline(x=iz,color=chcolor)
            hline_03 = ax[0,2].axhline(y=iy,color=chcolor)
            vline_04 = ax[0,3].axvline(x=self.t_axis[it], color=chcolor)
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
            ax[0,3].set_xlim((self.t_axis[0]*0.95,self.t_axis[-1]*1.05))
            ax[0,3].set_aspect('auto')

        def _on_click(event):

            if event.inaxes == ax[0,0]:
                self.coords[0] = event.ydata
                self.coords[1] = event.xdata
            elif event.inaxes == ax[0,1]:
                self.coords[0] = event.ydata
                self.coords[2] = event.xdata
            elif event.inaxes == ax[0,2]: 
                self.coords[1] = event.ydata
                self.coords[2] = event.xdata
            elif event.inaxes == ax[0,3]: 
                idx = np.abs(self.t_axis-event.xdata).argmin()
                self.coords[3] = idx
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
        self.coords = [i//2 for i in self.image.shape[:3]] + [0]
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
        _draw_param_slices()
        _draw_crosshairs()
        _draw_best_fit()

        fig.canvas.mpl_connect('button_press_event', _on_click)
        plt.show()
