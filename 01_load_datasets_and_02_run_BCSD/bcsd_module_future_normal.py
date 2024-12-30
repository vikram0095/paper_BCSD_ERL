from tqdm import tqdm
import scipy.stats
import numpy as np
import utility_module_2_v_3 as uu


def normal_quantile_mapping(var_data_obs,var_data_hist,var_data_pred):
    '''
    -> bias correction of a univariate time series
    -> does not care about daily/ monthly
    '''
    bias_corr=np.zeros(var_data_hist.shape[0]);

    mu,sig = scipy.stats.norm.fit(var_data_hist)

    cdf = scipy.stats.norm.cdf(var_data_pred,loc=mu,scale=sig)

    omu,osig = scipy.stats.norm.fit(var_data_obs)

    bias_corr=scipy.stats.norm.ppf(cdf,loc=omu,scale=osig)
    
    print("Nans",np.sum(np.isnan(bias_corr)))
    print("Infs",np.sum(np.isinf(bias_corr)))
    print("CDf==1 ,mu and sig",np.sum(cdf == 1),omu,osig)
    print("Values ",var_data_pred[cdf == 1])

    
    
    return bias_corr


def gamma_quantile_mapping(var_data_obs,var_data_hist_train,var_data_hist_pred):
    '''
    -> bias correction of a univariate time series
    -> does not care about daily/ monthly
    '''
    
    data=var_data_hist_train
    data_non_zeros=(data[data>=1])
    count_zeros=(data[data<0].shape[0])
    count_total=(data.shape[0])
    p_zeros=count_zeros/count_total
    fita,fitloc,fitscale = scipy.stats.gamma.fit(data_non_zeros,floc=0)
    
    data=var_data_hist_pred
    index_zeros = (data<1)
    data[index_zeros] = 1
    cdf= p_zeros + (1 - p_zeros) * scipy.stats.gamma.cdf(data, a=fita,loc=fitloc,scale=fitscale)
    cdf[index_zeros] = p_zeros
    
    data=var_data_obs
    data_non_zeros=(data[data>=1])
    count_zeros=(data[data<0].shape[0])
    count_total=(data.shape[0])
    p_zeros_imd=count_zeros/count_total
    ofita,ofitloc,ofitscale = scipy.stats.gamma.fit(data_non_zeros,floc=0)

    
    bias_corrected=np.zeros_like(var_data_hist_pred)
    for itr in range(cdf.shape[0]):
        cdfi=cdf[itr]
        if cdfi<=p_zeros_imd:
            bias_corrected[itr]=0
        else:
            z=(cdfi-p_zeros_imd)/(1-p_zeros_imd)
            bc=scipy.stats.gamma.ppf(z, a=ofita,loc=ofitloc,scale=ofitscale)
            bias_corrected[itr]=bc   
            
            
    # cdf[cdf <= p_zeros_imd] = np.nan
    # bias_corrected=scipy.stats.gamma.ppf((cdf-p_zeros_imd)/(1-p_zeros_imd), a=ofita,loc=ofitloc,scale=ofitscale)
    # bias_corrected[cdf <= p_zeros_imd] = 0
    return bias_corrected


def empirical_quantile_mapping(var_data_obs,var_data_hist_train,var_data_hist_pred,nbins=20):

    hist = np.histogram(var_data_hist_train, bins=nbins)
    hist_dist = scipy.stats.rv_histogram(hist)

    cdf = hist_dist.cdf(var_data_hist_pred)
    
    histo = np.histogram(var_data_obs, bins=nbins)
    hist_disto = scipy.stats.rv_histogram(histo)

    bias_corr=hist_disto.ppf(cdf)

    return bias_corr


def bias_corr_spatial_daily(data_obs_coarse,data_gcm_train,data_gcm_pred,dic,bias_correction_function):
    mask=dic['mask']
    lat_obs=dic['lat_obs']
    lat_gcm=dic['lat_gcm']
    lon_obs=dic['lon_obs']
    lon_gcm=dic['lon_gcm'] 
    
    N1g=lat_gcm.shape[0]
    N2g=lon_gcm.shape[0]
    
    print("Bias Correction Spatially - daily:")
    print("Training years :", data_obs_coarse.shape[0]/365 , '==' , data_obs_coarse.shape[0]/365 )
    print("Testing years :", data_gcm_pred.shape[0]/365)
    print("Bias correction fucntion :", bias_correction_function)
    print("Input shape",data_obs_coarse.shape,data_gcm_train.shape,data_gcm_pred.shape)

    data_obs_m=data_obs_coarse.flatten(order='F').reshape((365,-1,N1g,N2g),order='F') # DOYxNxN1xN2
    data_gcm_train_m=data_gcm_train.flatten(order='F').reshape((365,-1,N1g,N2g),order='F')
    data_gcm_pred_m=data_gcm_pred.flatten(order='F').reshape((365,-1,N1g,N2g),order='F')
    del data_obs_coarse
    del data_gcm_train
    del data_gcm_pred

    # print('Reshaped',data_obs_m.shape,data_gcm_train_m.shape,data_gcm_pred_m.shape)

    data_obs_m_ex=np.concatenate((data_obs_m[-15:,:,:,:],data_obs_m[:,:,:,:],data_obs_m[:15,:,:,:]),axis=0)
    data_gcm_train_m_ex=np.concatenate((data_gcm_train_m[-15:,:,:,:],data_gcm_train_m[:,:,:,:],data_gcm_train_m[:15,:,:,:]),axis=0)

    # print('Appended',data_obs_m_ex.shape,data_gcm_train_m_ex.shape)
    data_bc_gcm=np.empty_like(data_gcm_pred_m)

    del data_obs_m
    del data_gcm_train_m
    

    print(N1g,N2g)
    for i in tqdm(range(N1g)):
        for j in range(N2g):
            # if mask[i,j] == False:
                # print(i,j)
                for day_iter in range(365):
                    va=data_obs_m_ex[day_iter:day_iter+31,:,i,j].flatten(order='F')
                    vb=data_gcm_train_m_ex[day_iter:day_iter+31,:,i,j].flatten(order='F')
                    vc=data_gcm_pred_m[day_iter,:,i,j].flatten(order='F')

                    data_bc_gcm[day_iter,:,i,j]=bias_correction_function(va,vb,vc)

    ret =  data_bc_gcm.flatten(order='F').reshape((-1,N1g,N2g),order='F')
    print("Output shape",ret.shape)

    return ret



def bias_corr_spatial_monthly(data_obs_coarse,data_gcm_train,data_gcm_pred,dic ,bias_correction_function):
    
    mask=dic['mask']
    lat_gcm=dic['lat_gcm']
    lon_gcm=dic['lon_gcm'] 
    N1g=lat_gcm.shape[0]
    N2g=lon_gcm.shape[0]    
    print("Bias Correction Spatially - monthly:")
    print("Training years :", data_obs_coarse.shape[0]/12 , '==' , data_obs_coarse.shape[0]/12 )
    print("Testing years :", data_gcm_pred.shape[0]/12)
    print("Bias correction fucntion :", bias_correction_function)
    print("BC Input shape",data_obs_coarse.shape,data_gcm_train.shape,data_gcm_pred.shape)

    
    data_obs_coarse=data_obs_coarse.flatten(order='F').reshape((12,-1,N1g,N2g),order='F')
    data_gcm_train=data_gcm_train.flatten(order='F').reshape((12,-1,N1g,N2g),order='F')
    data_gcm_pred=data_gcm_pred.flatten(order='F').reshape((12,-1,N1g,N2g),order='F')

    data_bc_gcm=np.zeros(data_gcm_pred.shape)

    print(N1g,N2g)
    for i in tqdm(range(N1g)):
        for j in range(N2g):
            for month_iter in range(12):
                va=data_obs_coarse[month_iter,:,i,j].flatten()
                vb=data_gcm_train[month_iter,:,i,j].flatten()
                vc=data_gcm_pred[month_iter,:,i,j].flatten()
                data_bc_gcm[month_iter,:,i,j] = bias_correction_function(va,vb,vc).flatten()


    ret =  data_bc_gcm.flatten(order='F').reshape((-1,N1g,N2g),order='F')
    print("BC Output shape",ret.shape)
    return ret
  
    
def spatial_diaggregation(data_obs_train,data_bc_gcm_pred,dic,sd_type = "PREC",temporal_res = "Daily"):
    if temporal_res == "Daily":
        N_clim =365
    else:
        N_clim = 12
        
        

    lat_obs=dic['lat_obs']
    lat_gcm=dic['lat_gcm']
    lon_obs=dic['lon_obs']
    lon_gcm=dic['lon_gcm'] 
    
    N1=lat_obs.shape[0]
    N2=lon_obs.shape[0]

    N1g=lat_gcm.shape[0]
    N2g=lon_gcm.shape[0]
    
    N=data_bc_gcm_pred.shape[0]
    
    print("Spatial Disaggregation - " + temporal_res ,"(NCLIM == ",N_clim ,")")
    print("SD Type - " + sd_type)
    print("No of timesteps " , N)
    print("SD Input shape",data_obs_train.shape,data_bc_gcm_pred.shape)
    data_return=np.zeros((N, N1, N2))
    
    CLIM_fine=data_obs_train.reshape((N_clim,-1,N1,N2),order='F').mean(axis=1)
    CLIM_coarse = np.zeros((N_clim,N1g,N2g))
    
    ind_zeroes = CLIM_fine <1
    CLIM_fine[CLIM_fine <1 ] = 1
    for i in range(N_clim):
        CLIM_coarse[i,:,:]=uu.regrid(CLIM_fine[i,:,:],lat_obs,lon_obs,lat_gcm,lon_gcm)
    
    for i in tqdm(range(N)):
        if sd_type == "PREC":
            delta_F= data_bc_gcm_pred[i,:,:]/CLIM_coarse[i%N_clim]            
            data_fined_bc_gcm=uu.regrid(delta_F,lat_gcm,lon_gcm,lat_obs,lon_obs)
            data_fined_bc_gcm[ind_zeroes[i%N_clim,:,:]] = 0
            data_return[i,:,:]=data_fined_bc_gcm * CLIM_fine[i%N_clim]
            
        else:
            data_fined_bc_gcm=uu.regrid(data_bc_gcm_pred[i,:,:]-CLIM_coarse[i%N_clim],
                                        lat_gcm,lon_gcm,lat_obs,lon_obs)
            data_fined_bc_gcm[np.isinf(data_fined_bc_gcm)] = 0  
            data_return[i,:,:]=data_fined_bc_gcm + CLIM_fine[i%N_clim]

    print("SD Output shape",data_return.shape)
    return data_return


def bcsd(experiment_id,data_obs_train_gridded,data_gcm_train_gridded,data_gcm_test_gridded,
         metadata,bias_correction_function,temporal_res = "Daily",sd_type ="PREC",mode=1):
    
    Nyears_train=metadata['Nyears_train']
    Nyears_test=metadata['Nyears_test']
    mask=metadata['mask']
    lat_obs=metadata['lat_obs']
    lat_gcm=metadata['lat_gcm']
    lon_obs=metadata['lon_obs']
    lon_gcm=metadata['lon_gcm'] 
    path_out = metadata['path_out'] 
                        
    data_obs_train_coarse=np.zeros((data_obs_train_gridded.shape[0],data_gcm_train_gridded.shape[1],data_gcm_train_gridded.shape[2]))
    for i in range(data_obs_train_gridded.shape[0]):
        data_obs_train_coarse[i,:,:]=uu.regrid(data_obs_train_gridded[i,:,:],lat_obs,lon_obs,lat_gcm,lon_gcm)
                
                         
    if mode ==1:
        if temporal_res == "Monthly":
            data_gcm_bc_pred=bias_corr_spatial_monthly(data_obs_train_coarse,data_gcm_train_gridded,
                                                       data_gcm_test_gridded,metadata,bias_correction_function)
        else:
            data_gcm_bc_pred=bias_corr_spatial_daily(data_obs_train_coarse,data_gcm_train_gridded,
                                                       data_gcm_test_gridded,metadata,bias_correction_function)

        np.save(path_out + '/future_normal_BC_outputs_'+experiment_id,data_gcm_bc_pred)
    else:
        data_gcm_bc_pred=np.load(path_out + '/future_normal_BC_outputs_'+experiment_id+'.npy')
                         
    data_bcsd_pred=spatial_diaggregation(data_obs_train_gridded,data_gcm_bc_pred,metadata,temporal_res =temporal_res,sd_type = "PREC")
    np.save(path_out + '/future_normal_BCSD_outputs_'+experiment_id,data_bcsd_pred)

    return data_bcsd_pred


