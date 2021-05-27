import xarray as xr
import numpy as np

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
        self._obj.orientation[:,:,0]=np.mod(np.pi-self._obj.orientation[:,:,0],2*np.pi)
        
    def flipud(self):
        '''
        flip up down the data and rotate the orientation 
        '''
        self._obj.coords['y']=np.max(self._obj.coords['y'])-self._obj.coords['y']
        self._obj.orientation[:,:,0]=np.mod(2*np.pi-self._obj.orientation[:,:,0],2*np.pi)
        
    def rot180(self):
        '''
        rotate 180 degre the data and rotate the orientation 
        '''
        self._obj.coords['x']=np.max(self._obj.coords['x'])-self._obj.coords['x']
        self._obj.coords['y']=np.max(self._obj.coords['y'])-self._obj.coords['y']
        self._obj.orientation[:,:,0]=np.mod(np.pi+self._obj.orientation[:,:,0],2*np.pi)
        
        
#--------------------geometric transformation---------------------------
    def filter(self,val):
        '''
        Put nan value in orientation file
        '''
        idx,idy=np.where(self._obj.quality<val)
        
        new=np.array(self._obj.orientation)
        new[idx,idy,:]=np.nan
        self._obj.orientation[:,:,:]=new           
