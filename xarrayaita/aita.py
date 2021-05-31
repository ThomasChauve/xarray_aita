import xarray as xr
import xarrayuvecs.uvecs as xu
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
import ipywidgets as widgets

@xr.register_dataset_accessor("aita")

class aita(object):
    '''
    This is a classe to work on aita data in xarray environnement.
    
    .. note:: xarray does not support heritage from xr.DataArray may be the day it support it, we could move to it
    '''
    
    def __init__(self, xarray_obj):
        '''
        Constructor for aita. 
        
        The xarray_obj should contained at least :
        1. orientation : DataArray that is compatible with uvec structure
        2. quality : DataArray of dim (m,n,1)
        
        It can contained :
        1. micro : DataArray of dim (m,n,1)
        2. grainId : DataArray of dim (m,n,1)
        
        :param xarray_obj:
        :type xarray_obj: xr.DataArray
        '''
        self._obj = xarray_obj 
    pass
#--------------------geometric transformation---------------------------
    def fliplr(self):
        '''
        flip left right the data and rotate the orientation 
        '''
        self._obj.coords['x']=np.max(self._obj.coords['x'])-self._obj.coords['x']
        self._obj.orientation[:,:,0]=np.mod(2*np.pi-self._obj.orientation[:,:,0],2*np.pi)

        
    def rot180(self):
        '''
        rotate 180 degre the data and rotate the orientation 
        '''
        self._obj.coords['x']=np.max(self._obj.coords['x'])-self._obj.coords['x']
        self._obj.coords['y']=np.max(self._obj.coords['y'])-self._obj.coords['y']
        self._obj.orientation[:,:,0]=np.mod(np.pi+self._obj.orientation[:,:,0],2*np.pi)
        
        
#--------------------function---------------------------
    def filter(self,val):
        '''
        Put nan value in orientation file
        '''
        idx,idy=np.where(self._obj.quality<val)
        
        new=np.array(self._obj.orientation)
        new[idx,idy,:]=np.nan
        self._obj.orientation[:,:,:]=new
        
        
    def crop(self,lim,rebuild_gId=True):
        '''
        :param rebuild_gId: recompute the grainID
        :type rebuild_gId: bool
        :param lim:
        :type lim: np.array
        '''
        ds=self._obj.where((self._obj.x>np.min(lim[0])) * (self._obj.x<np.max(lim[0])) *(self._obj.y>np.min(lim[1]))*(self._obj.y<np.max(lim[1])),drop=True)
        
        if rebuild_gId:
            ds.grainId.data=morphology.label(ds.micro, connectivity=1, background=1)
        
        return ds
#---------------interactive function-------------------
    def interactive_crop(self,rebuild_gId=True):
        '''
        out=data.aita.interactive_crop()
        
        :param rebuild_gId: recompute the grainID
        :type rebuild_gId: bool

        This function can be use to crop within a jupyter notebook
        It will crop the data and export the value of the crop in out.pos
        '''
        
        def onselect(eclick, erelease):
            "eclick and erelease are matplotlib events at press and release."
            print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
            print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
            print('used button  : ', eclick.button)

        def toggle_selector(event):
            print('Key pressed.')
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                print('RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                print('RectangleSelector activated.')
                toggle_selector.RS.set_active(True)

        print('1. click and drag the mouse on the figure to selecte the area')
        print('2. you can draw the rectangle using the button "Draw area"')
        print('3. if you are unhappy with the selection restart to 1.')
        print('4. if you are happy with the selection click on "Export crop" (only the last rectangle is taken into account)')

        
        fig,ax=plt.subplots()
        ml=np.max(np.array([len(self._obj.x),len(self._obj.y)]))
        fig.set_figheight(len(self._obj.y)/ml*15)
        fig.set_figwidth(len(self._obj.x)/ml*15)
        tmp=self._obj.copy()
        tmp['colormap']=self._obj.orientation.uvecs.calc_colormap()
        tmp.colormap.plot.imshow()
        toggle_selector.RS = matplotlib.widgets.RectangleSelector(ax, onselect, drawtype='box')
        fig.canvas.mpl_connect('key_press_event', toggle_selector)


        buttonCrop = widgets.Button(description='Export crop')
        buttonDraw = widgets.Button(description='Draw area')
        
        def draw_area(_):
            x=list(toggle_selector.RS.corners[0])
            x.append(x[0])
            y=list(toggle_selector.RS.corners[1])
            y.append(y[0])
            plt.plot(x,y,'-k')

        def get_data(_):
            x=list(toggle_selector.RS.corners[0])
            x.append(x[0])
            y=list(toggle_selector.RS.corners[1])
            y.append(y[0])
            plt.plot(x,y,'-b')
            
            # what happens when we press the button
            out=self.crop(lim=toggle_selector.RS.corners,rebuild_gId=rebuild_gId)
            
            get_data.ds=out
            get_data.crop=toggle_selector.RS.corners
                
            return get_data
            
            


        # linking button and function together using a button's method
        buttonDraw.on_click(draw_area)
        buttonCrop.on_click(get_data)
        # displaying button and its output together
        display(buttonDraw,buttonCrop)

        return get_data

    def interactive_misorientation_profile(self,res=0,degre=True):
        '''
        Interactive misorientation profile for jupyter notebook
        
        :param res: step size of the profil
        :type res: float
        :param degre: return mis2o and mis2p in degre overwise in radian (default: true)
        :type degre: bool
        '''
        fig,ax=plt.subplots()
        ml=np.max(np.array([len(self._obj.x),len(self._obj.y)]))
        fig.set_figheight(len(self._obj.y)/ml*15)
        fig.set_figwidth(len(self._obj.x)/ml*15)
        tmp=self._obj.copy()
        tmp['colormap']=tmp.orientation.uvecs.calc_colormap()
        tmp.colormap.plot.imshow()
        ax.axis('equal')

        pos = []
        def onclick(event):
            pos.append([event.xdata,event.ydata])
            if len(pos)==1:
                plt.plot(pos[0][0],pos[0][1],'sk')
            else:
                pos_mis=np.array(pos[-2::])
                plt.plot(pos[-2][0],pos[-2][1],'sk')
                plt.plot(pos[-1][0],pos[-1][1],'ok')
                plt.plot(pos_mis[:,0],pos_mis[:,1],'-k')


        fig.canvas.mpl_connect('button_press_event', onclick)

        buttonShow = widgets.Button(description='Show line')
        buttonExtract = widgets.Button(description='Extract profile')


        def extract_data(_):
            pos_mis=np.array(pos[-2::])
            ll=((pos_mis[1,1]-pos_mis[1,0])**2+(pos_mis[0,1]-pos_mis[0,0])**2)**0.5
            if res==0:
                rr=np.array(np.abs(self._obj.x[1]-self._obj.x[0]))
            else:
                rr=res
            nb=int(ll/rr)
            
            xx=np.linspace(pos_mis[0,0],pos_mis[1,0],nb)
            yy=np.linspace(pos_mis[0,1],pos_mis[1,1],nb)
            ds,vxyz=self._obj.orientation.uvecs.misorientation_profile(xx,yy,degre=degre)
            ds.attrs["start"]=pos_mis[0,:]
            ds.attrs["end"]=pos_mis[1,:]
            ds.attrs["step_size"]=rr
            ds.attrs["unit"]=self._obj.unit

            extract_data.ds=ds
            extract_data.vxyz=vxyz

            return extract_data


        # linking button and function together using a button's method
        buttonExtract.on_click(extract_data)

        # displaying button and its output together
        display(buttonExtract)

        return extract_data