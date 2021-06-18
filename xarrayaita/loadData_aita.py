import xarray as xr
import pandas as pd
import numpy as np
from skimage import io
from skimage import morphology
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import scipy

def aita5col(adr_data,micro_adress=0):
    '''
    Function to load the data from G50 analyser that have 5 columns
    
    :param adr_data: path to the data
    :type adr_data: str
    :param micro_adress: path to microstructure file (black and white image where grains boundaries are white) `.bmp`
    :type micro_adress: str
    '''
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
    
    dx=np.linspace(0,nx-1,nx)*resolution
    dy=np.linspace(0,ny-1,ny)*resolution
    dy=np.max(dy)-dy
    #-------------------- The data structure--------------------------
    ds = xr.Dataset(
    {   
        "orientation": (["y", "x","uvecs"],(np.dstack( (np.array(data.azi*np.pi/180).reshape([ny,nx]),np.array(data.col*np.pi/180).reshape([ny,nx])) ))),
        "quality": (["y", "x"],np.array(data.qua).reshape([ny,nx])),
        "micro": (["y", "x"],micro_field),
        "grainId": (["y", "x"], morphology.label(micro_field, connectivity=1, background=1)),
        
    },
    coords={
        "x": dx,
        "y": dy,
    },
    )
    
    ds.attrs["date"]=a[3][5:-1]
    ds.attrs["unit"]='millimeters'
    ds.attrs["step_size"]=resolution
    ds.attrs["path_dat"]=adr_data
    return ds


def aita3col(adr_data,image,micro_adress=0):
    '''
    Function to load the data from G50 analyser that have 3 columns
    
    :param adr_data: path to the data
    :type adr_data: str
    :param image: path to one image from G50 analyser
    :type image: str
    :param micro_adress: path to microstructure file (black and white image where grains boundaries are white) `.bmp`
    :type micro_adress: str
    '''
    data=pd.read_csv(adr_data,usecols=[0,1,2],skiprows=13,comment='[',header=0,names=['azi','col','qua'],delimiter=' ')
    
    # read head of file
    file=open(adr_data,'r')
    a=[]
    [a.append(file.readline()) for i in list(range(16))]
    file.close()
    # resolution mu m
    res=int(a[5][10:12])
    # transforme the resolution in mm
    resolution=res/1000. 
    
    im = io.imread(image)
    ss=im.shape    
    # number of pixel along x
    nx=ss[1]
    # number of pixel along y
    ny=ss[0]
    
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
    
    dx=np.linspace(0,nx-1,nx)*resolution
    dy=np.linspace(0,ny-1,ny)*resolution
    dy=np.max(dy)-dy
    #-------------------- The data structure--------------------------
    ds = xr.Dataset(
    {   
        "orientation": (["y", "x","uvecs"],(np.dstack( (np.array(data.azi*np.pi/180).reshape([ny,nx]),np.array(data.col*np.pi/180).reshape([ny,nx])) ))),
        "quality": (["y", "x"],np.array(data.qua).reshape([ny,nx])),
        "micro": (["y", "x"],micro_field),
        "grainId": (["y", "x"], morphology.label(micro_field, connectivity=1, background=1)),
        
    },
    coords={
        "x": dx,
        "y": dy,
    },
    )
    
    ds.attrs["date"]=a[3][5:-1]
    ds.attrs["unit"]='millimeters'
    ds.attrs["step_size"]=resolution
    ds.attrs["path_dat"]=adr_data
    return ds

def craft_input(adr_vtk,adr_phase):
    '''
    Function to load the data from craft simulation
    
    :param adr_vtk: path to the vtk
    :type adr_data: str
    :param phase: path to phase
    :type phase: str
    '''
    reader = vtk.vtkDataSetReader() 
    reader.SetFileName(adr_vtk)
    reader.Update()
    ug  = reader.GetOutput()
    res=ug.GetSpacing()[0]
    ugdim=ug.GetDimensions()
    map=vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((ug.GetDimensions()[0:2][::-1]))
    
    orientation=np.transpose(np.loadtxt(adr_phase,unpack=True, skiprows=9, usecols=(0,2,3,4),dtype='f,f,f,f'))
    
    phi1=np.zeros(map.shape)
    phi=np.zeros(map.shape)
    
    for i in np.unique(map):
        phi1[map==i]=orientation[orientation[:,0]==i,1][0]
        phi[map==i]=orientation[orientation[:,0]==i,2][0]

    azi=np.mod(phi1-np.pi/2,2*np.pi)
    
    micro=(scipy.signal.convolve2d(map,np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]))>0)[1:-1,1:-1]
    
    ss=map.shape
    
    dy=np.linspace(0,ss[0]-1,ss[0])*res
    dx=np.linspace(0,ss[1]-1,ss[1])*res
    
    ds = xr.Dataset(
    {   
        "orientation": (["y", "x","uvecs"],np.dstack((azi,phi))),
        "quality": (["y", "x"],np.ones(azi.shape)*100),
        "micro": (["y", "x"],micro),
        "grainId": (["y", "x"], map),
        
    },
    coords={
        "x": dx,
        "y": dy,
    },
    )
    
    ds.attrs["unit"]='millimeters'
    ds.attrs["step_size"]=res
    ds.attrs["path_dat"]=adr_vtk
                    
    return ds