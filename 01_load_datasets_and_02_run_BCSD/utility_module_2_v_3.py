

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from numpy import meshgrid
from netCDF4 import Dataset
import pandas as pd





def find_bounds(vector,lower,upper,v=0):
  # chops an ascending vector when lower bound and upper bound are given.
  # returns start and end indices as well as chopped vector
  # useful for cropping spatial datasets
  switch_l=0
  switch_u=0
  
  if lower < vector[0]:
    switch_l=1
    ind_low=0
    if v==1:
      print("find_bounds: Lower is less than first value in vector")


  if upper > vector[-1]:
    switch_u=1
    ind_upp=vector.shape[0]-1
    if v==1:
      print("find_bounds: Upper is greater than last value in vector")


  if lower<=upper:
    for iter in range(vector.shape[0]):
      if  vector[iter]> lower and switch_l==0:
        #print(vector[iter-1])
        ind_low=iter-1
        switch_l=1
      elif  vector[iter]== lower and switch_l==0:
        #print(vector[iter-1])
        ind_low=iter
        switch_l=1
      if vector[iter]>= upper and switch_u==0:
        #print(vector[iter])
        ind_upp=iter
        switch_u=1
      
  else:
    ind_low=-9999
    ind_upp=-9999
    if v==1:
      print("find_bounds: Lower is greater than upper")
  return (ind_low,ind_upp,vector[ind_low:ind_upp+1])
  
def find_bounds_dec(vector,upper,lower):

  l,u,_=find_bounds((vector[::-1]),lower,upper)
  if l<-9998:
    ind_low=l
    ind_upp=u
  else:
    ind_low=vector.shape[0]-l-1
    ind_upp=vector.shape[0]-u-1

  return (ind_upp,ind_low,vector[ind_upp:ind_low+1])


def isleapyear(yeaR):
  if (( yeaR%400 == 0)or (( yeaR%4 == 0 ) and ( yeaR%100 != 0))):
      return 1
  else:
      return 0


def box_to_cen(box):
  return (box[1:]+box[:-1])*0.5

def cen_to_box(cen):
  box=np.zeros(cen.shape[0]+1)
  box[0]=cen[0]+0.5*(cen[0]-cen[1])
  box[1:-1]=(cen[1:]+cen[:-1])*0.5
  box[-1]=cen[-1]+0.5*(cen[-1]-cen[-2])
  return box




def map_plot_cyl(data_lat_x_lon,lat_cen,lon_cen,map_bounds,mp_spacing=10):
  

  lat_box=cen_to_box(lat_cen)
  lon_box=cen_to_box(lon_cen)

  fig=plt.figure(figsize=(15,7))
  map = Basemap(projection='cyl',resolution='l',
                llcrnrlon=map_bounds[0], 
                urcrnrlat=map_bounds[1],
                urcrnrlon=map_bounds[2], 
                llcrnrlat=map_bounds[3])


  xx, yy = meshgrid(lon_box,lat_box )
  #return  dimenion of lat * lon

  map.pcolor(xx, yy, data_lat_x_lon,cmap='jet')
  map.drawmeridians(np.arange(-180,180,mp_spacing),labels=[0,0,0,1], linewidth=1.0) #longitudes
  map.drawparallels(np.arange(-90,90,mp_spacing),labels=[1,0,0,0], linewidth=1.0) #latitudes

  map.drawcountries(linewidth=1)
  map.drawcoastlines(linewidth=1)

  map.colorbar()

  plt.show()
  
def get_filename_ncar(path,yeaR):

  fname = path + str(yeaR)+'.nc'
  return fname

def extract_ncar(st,en,var_nc_name,path,data_bounds):
  data_daily=np.array([])

  for yeaR in range(st,en+1):
    step=365+isleapyear(yeaR)
    filename = get_filename_ncar(path,yeaR)
    
    fh = Dataset(filename, mode='r')
    lons = fh.variables['lon'][:]
    lats = fh.variables['lat'][:]
    data = fh.variables[var_nc_name][:]

    l_lat,u_lat,lat_new=find_bounds_dec(lats,data_bounds[1],data_bounds[3])
    l_lon,u_lon,lon_new=find_bounds(lons,data_bounds[0],data_bounds[2])

    data_cropped=data[:,l_lat:u_lat+1,l_lon:u_lon+1]
    
    if yeaR==st:
      data_daily=data_cropped
    else:
      data_daily=np.concatenate((data_daily,data_cropped),axis=0)
  return data_daily,lat_new,lon_new



def extract_imd_rainfall_nc(st,en,path,data_bounds):
  data_daily=np.array([])

  for yeaR in range(st,en+1):
    step=365+isleapyear(yeaR)

    rain_nc_file = path + "rainfall_"+str(yeaR)+".nc"
    fh = Dataset(rain_nc_file, mode='r')
    print(fh.variables.keys())
    if 'LONGITUDE' in  fh.variables.keys():
        lons = fh.variables['LONGITUDE'][:]
        lats = fh.variables['LATITUDE'][:]
        data_rf = fh.variables['RAINFALL'][:]#time lat lon
    else:
        print(fh.variables.keys())
        lons = fh.variables['lon'][:]
        lats = fh.variables['lat'][:]
        data_rf = fh.variables['rainfall'][:]#time lat lon

        

    if yeaR==st:
      mask=data_rf.mask

    l_lat,u_lat,lat_new=find_bounds(lats,data_bounds[3],data_bounds[1])
    l_lon,u_lon,lon_new=find_bounds(lons,data_bounds[0],data_bounds[2])

    data_cropped=data_rf[:,l_lat:u_lat+1,l_lon:u_lon+1]
    
    if yeaR==st:
      data_daily=data_cropped
    else:
      data_daily=np.concatenate((data_daily,data_cropped),axis=0)

  data_daily[data_daily<=-99]=np.nan

  return data_daily,mask,lat_new,lon_new



def extract_imd_temp(st,en,path,data_bounds):
  data_daily=np.array([])

  for yeaR in range(st,en+1):
    step=365+isleapyear(yeaR)

    data=np.fromfile(path+str(yeaR)+".grd", dtype='float32').reshape(step,31,31)

    lats=np.arange(7.5,37.6,1)
    lons=np.arange(67.5,97.6,1)

    l_lat,u_lat,lat_new=find_bounds(lats,data_bounds[3],data_bounds[1])
    l_lon,u_lon,lon_new=find_bounds(lons,data_bounds[0],data_bounds[2])

    data_cropped=data[:,l_lat:u_lat+1,l_lon:u_lon+1]
    
    if yeaR==st:
      data_daily=data_cropped
    else:
      data_daily=np.concatenate((data_daily,data_cropped),axis=0)
  
  data_daily[data_daily<=-99]=np.nan
  data_daily[data_daily>=99]=np.nan

  return data_daily,lat_new,lon_new



def extract_imd_rainfall_pt25(st,en,path,data_bounds):
  data_daily=np.array([])

  for yeaR in range(st,en+1):
    step=365+isleapyear(yeaR)

    data=np.fromfile(path + "rainfall_"+str(yeaR)+".grd", dtype='float32').reshape(step,129,135)


    lats=np.arange(6.5,38.5+.1,.25)
    lons=np.arange(66.5,100.0+.1,.25)

    l_lat,u_lat,lat_new=find_bounds(lats,data_bounds[3],data_bounds[1])
    l_lon,u_lon,lon_new=find_bounds(lons,data_bounds[0],data_bounds[2])

    data_cropped=data[:,l_lat:u_lat+1,l_lon:u_lon+1]
    
    if yeaR==st:
      data_daily=data_cropped
    else:
      data_daily=np.concatenate((data_daily,data_cropped),axis=0)
  
  data_daily[data_daily<-99]=np.nan

  return data_daily,lat_new,lon_new



def daily_to_monthly_sts(data_daily,st,en,method='SUM'):
  #data daily add new axis
  data_monthly=np.zeros(((en-st+1)*12,data_daily.shape[1],data_daily.shape[2]))
  print("Monthly data dimensions:",data_monthly.shape)
  year_index=0
  itex=0
  for yeaR in range(st,en+1):
    months=[0,31,28+isleapyear(yeaR),31,30,31,30,31,31,30,31,30,31]
    ind_months=(np.cumsum(months))
    #print(ind_months)
    for month_iter in range(12):
      itex=(yeaR-st)*12+month_iter
      if method=='SUM':
        data_monthly[itex,:,:]=np.nansum(data_daily[year_index+ind_months[month_iter]:year_index+ind_months[month_iter+1],:,:],axis=0)
      elif method=='MEAN':
        data_monthly[itex,:,:]=np.nanmean(data_daily[year_index+ind_months[month_iter]:year_index+ind_months[month_iter+1],:,:],axis=0)
      else:
        data_monthly[itex,:,:]=np 
    year_index=year_index+365+isleapyear(yeaR)
  return data_monthly

#three month aggregation
def monthly_to_trimonthly_sts(data_monthly,method='SUM'):
  L=int(data_monthly.shape[0]/12*4)
  data_trimonthly=np.zeros((L,data_monthly.shape[1],data_monthly.shape[2]))
  print("3-Monthly data dimensions:",data_trimonthly.shape)

  for iter in range(0,L):
    itex=iter*3
    if method=='SUM':
      data_trimonthly[iter,:,:]=np.nansum(data_monthly[itex:itex+3,:,:],axis=0)
    elif method=='MEAN':
      data_trimonthly[iter,:,:]=np.nanmean(data_monthly[itex:itex+3,:,:],axis=0)
  return data_trimonthly

#six month aggregation
def monthly_to_sixmonthly_sts(data_monthly,method='SUM'):
  L=int(data_monthly.shape[0]/12*2)
  data_sixmonthly=np.zeros((L,data_monthly.shape[1],data_monthly.shape[2]))
  print("6-Monthly data dimensions:",data_sixmonthly.shape)

  for iter in range(0,L):
    itex=iter*6
    if method=='SUM':
      data_sixmonthly[iter,:,:]=np.nansum(data_monthly[itex:itex+6,:,:],axis=0)
    elif method=='MEAN':
      data_sixmonthly[iter,:,:]=np.nanmean(data_monthly[itex:itex+6,:,:],axis=0)
  return data_sixmonthly

#12 month aggregation
def monthly_to_yearly_sts(data_monthly,method='SUM'):
  L=int(data_monthly.shape[0]/12)
  data_yearly=np.zeros((L,data_monthly.shape[1],data_monthly.shape[2]))
  print("12-Monthly data dimensions:",data_yearly.shape)

  for iter in range(0,L):
    itex=iter*12
    if method=='SUM':
      data_yearly[iter,:,:]=np.nansum(data_monthly[itex:itex+12,:,:],axis=0)
    elif method=='MEAN':
      data_yearly[iter,:,:]=np.nanmean(data_monthly[itex:itex+12,:,:],axis=0)
  return data_yearly
  
  

def extract_JJAS_sts(ts,st,en,tstype='POINT',f='D'):
  # requires numpy and pandas
  # tstype ='SPATIAL','POINT'
  # for spatial ts should be time  x lat X lon


  dti = pd.date_range( start=str(st)+'-01-01', end=str(en)+'-12-31', freq=f)
  #print(dti.shape,ts.shape)
  index_jjas= np.any(np.array([ dti.month==6,dti.month==7,dti.month==8,dti.month==9]),axis=0)
  
  date_JJAS=dti[index_jjas]

  if tstype=='SPATIAL':
    data_JJAS=ts[index_jjas,:,:]
  else:
    data_JJAS=ts[index_jjas]

  return data_JJAS,date_JJAS

def extract_non_JJAS_sts(ts,st,en,tstype='POINT',f='D'):
  # requires numpy and pandas
  # tstype ='SPATIAL','POINT'
  # for spatial ts should be time  x lat X lon


  dti = pd.date_range( start=str(st)+'-01-01', end=str(en)+'-12-31', freq=f)
  #print(dti.shape,ts.shape)
  index_non_jjas= np.any(np.array([ dti.month==1,dti.month==2,dti.month==3,dti.month==4,dti.month==5,dti.month==10,dti.month==11,dti.month==12]),axis=0)
  
  date_JJAS=dti[index_non_jjas]

  if tstype=='SPATIAL':
    data_JJAS=ts[index_non_jjas,:,:]
  else:
    data_JJAS=ts[index_non_jjas]

  return data_JJAS,date_JJAS


def extract_non_JJAS_chopped_sts(ts,st,en,tstype='POINT',f='D'):
  # requires numpy and pandas
  # tstype ='SPATIAL','POINT'
  # isleapyear required
  # for spatial ts should be time  x lat X lon

  if f=='D':
    data_JJAS,date_JJAS=extract_non_JJAS(ts,st,en,f='D')
    ind1=31+28+isleapyear(st)+31+30+31
    ind2=31+30+31
    data_JJAS=data_JJAS[ind1:-ind2]
    date_JJAS=date_JJAS[ind1:-ind2]

  elif f=='M':
    data_JJAS,date_JJAS=extract_non_JJAS(ts,st,en,f='M')
    data_JJAS=data_JJAS[5:-3]
    date_JJAS=date_JJAS[5:-3]

  return data_JJAS,date_JJAS

def remove_leap_years(daily_ts,st,en):
  dti = pd.date_range( start=str(st)+'-01-01', end=str(en)+'-12-31', freq='D')
  #print(dti.shape,ts.shape)
  
  index_not= np.logical_not(np.all(np.array([ dti.month==2,dti.day==29]),axis=0))
  date_1=dti[index_not]
  data_1=daily_ts[index_not]
  return data_1,date_1

def add_leap_years(daily_ts,st,en):

  dti = pd.date_range( start=str(st)+'-01-01', end=str(en)+'-12-31', freq='D')
  data_ret=np.zeros(dti.shape)

  index_not= np.logical_not(np.all(np.array([ dti.month==2,dti.day==29]),axis=0))
  data_ret[index_not]=daily_ts

  index_yup= np.arange(dti.shape[0])[np.logical_not(index_not)]  
  data_ret[index_yup]=data_ret[index_yup-1]

  #print("cc",index_yup,index_yup-1,data_ret[index_yup],data_ret[index_yup-1])
  return data_ret,dti



  
def anomalize_ts(ts,st,en):
  #print(ts.shape)
  ts_rm=utility_module_2.remove_leap_years(ts,st,en)[0]
  #print(ts_rm.shape)
  ts_clim=ts_rm.reshape([365,-1],order='F')
  #print(ts_clim.shape)

  ts_anom=np.divide((ts_clim-np.mean(ts_clim,axis=1)[:,np.newaxis]),np.std(ts_clim,axis=1)[:,np.newaxis]).flatten(order='F')
  #print(ts_anom.shape)

  ts_anom_final=utility_module_2.add_leap_years(ts_anom,st,en)[0]
  #print(ts_anom_final.shape)
  
  #plt.plot(ts_anom_final)
  return ts_anom_final

#anomalize_ts(data_out[2,:],st,en)
def get_climatology_daily_sts(daily_ts,st,en):
  #print(ts.shape)
  ts_rm=remove_leap_years_2d(daily_ts,st,en)[0]
  #print(ts_rm.shape)

  ts_clim=ts_rm.reshape([365,-1,ts_rm.shape[1],ts_rm.shape[2]],order='F')
  return np.mean(ts_clim,axis=1)

def get_climatology_monthly_sts(monthly_ts,st,en):

  ts_clim=monthly_ts.reshape([12,-1,ts_rm.shape[1],ts_rm.shape[2]],order='F')
  return np.mean(ts_clim,axis=1)
 
def remove_leap_years_sts(daily_ts,st,en):

  dti = pd.date_range( start=str(st)+'-01-01', end=str(en)+'-12-31', freq='D')
  #print(dti.shape,ts.shape)
  
  index_not= np.logical_not(np.all(np.array([ dti.month==2,dti.day==29]),axis=0))
  date_1=dti[index_not]
  data_1=daily_ts[index_not,:,:]
  return data_1,date_1

def add_leap_years_sts(daily_ts,st,en):
 
  dti = pd.date_range( start=str(st)+'-01-01', end=str(en)+'-12-31', freq='D')
  data_ret=np.zeros((dti.shape[0],daily_ts.shape[1],daily_ts.shape[2]))
 
  index_not= np.logical_not(np.all(np.array([ dti.month==2,dti.day==29]),axis=0))
  data_ret[index_not,:,:]=daily_ts
 
  index_yup= np.arange(dti.shape[0])[np.logical_not(index_not)]  
  data_ret[index_yup,:,:]=data_ret[index_yup-1,:,:]
 
  #print("cc",index_yup,index_yup-1,data_ret[index_yup],data_ret[index_yup-1])
  return data_ret,dti

def regrid(z,lat_old,lon_old,lat_new,lon_new):
  from scipy import interpolate
  y=lat_new
  x=lon_new
  #xx,yy=np.meshgrid(lon_new, lat_new)

  f  = interpolate.interp2d(lon_old, lat_old, z, kind='linear')

  znew = f(lon_new, lat_new)
  return znew
  
  
def correct_nan_inf_neg(data_prec):
  data_prec[data_prec<0]=0
  data_prec[np.isnan(data_prec)]=0
  data_prec[np.isinf(data_prec)]=0  
  
  return data_prec 