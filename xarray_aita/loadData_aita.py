import xarray as xr
import pandas as pd
import numpy as np
from skimage import io
from skimage import morphology

def aita5col(adr_data,micro_adress=0):
    data=pd.read_csv(adr_data,usecols=[1,2,3,4,6],skiprows=16,comment='[',header=0,names=['x','y','azi','col','qua'],delimiter=' ')
    
    # read head of file
    file=open(adr_data,'r')
    a=[]
    [a.append(file.readline()) for i in list(range(16))]
    file.close()
    # resolution mu m
    res=int(a[5][10:12])
    # transforme the resolution in mm
    resolution=res/1000. 
    # number of pixel along x
    nx=int(a[14][9:14])
    # number of pixel along y
    ny=int(a[15][9:13])
    
    # microstrucure
    #open micro.bmp if necessary
    if micro_adress!=0:
        micro_bmp = io.imread(micro_adress)
        mm=np.max(micro_bmp)
        if len(micro_bmp.shape)==3:
            micro_field=micro_bmp[:,:,0]/mm
        elif len(micro_bmp.shape)==2:
            micro_field=micro_bmp[:,:]/mm
    else:
        micro_field=np.zeros([ny,nx])
    

    #-------------------- The data structure--------------------------
    ds = xr.Dataset(
    #"dims": {'x':resolution,'y':resolution}
    {   
        "orientation": (["y", "x","v"],(np.dstack( (np.array(data.azi*np.pi/180).reshape([ny,nx]),np.array(data.col*np.pi/180).reshape([ny,nx])) ))),
        "quality": (["y", "x"],np.array(data.qua).reshape([ny,nx])),
        "micro": (["y", "x"],micro_field),
        "grainId": (["y", "x"], morphology.label(micro_field, connectivity=1, background=1)),
        
    },
    coords={
        "x": np.linspace(0,nx-1,nx)*resolution,
        "y": np.linspace(0,ny-1,ny)*resolution,
    },
    )
    
    return ds
