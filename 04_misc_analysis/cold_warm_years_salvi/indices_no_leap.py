import numpy as np
import scipy.stats
def mean_sp(daily_ts):
  
  return daily_ts.mean(axis=0)


def mean_sp_annualy(daily_ts,st,en):
  CDD_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    CDD_year[yeaR]=mean_sp(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return CDD_year


def mean_sp_annualy_ly(daily_ts,st,en):
  CDD_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    CDD_year[yeaR]=mean_sp(daily_year_ts)
    days_iter=days_iter+365
  return CDD_year




def CDD(daily_ts):
  N=daily_ts.shape[0]
  #print(N)
  cdd_max=0
  i=0
  while(True):
    if i>=N:
      break
    else:
      cdd=0
      #print("outer")
      while(True):
        
        if daily_ts[i]<1 :
          cdd=cdd+1
          #print(cdd,i,daily_ts[i])
          i=i+1
          if i+1>N:
            break;
        else:
          #print(-1,i,daily_ts[i])
          i=i+1
          break
      cdd_max=max(cdd, cdd_max)
      
  return cdd_max


def CDD_annualy(daily_ts,st,en):
  CDD_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    CDD_year[yeaR]=CDD(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return CDD_year

def CDD_annualy_ly(daily_ts,st,en):
  CDD_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    CDD_year[yeaR]=CDD(daily_year_ts)
    days_iter=days_iter+365
  return CDD_year


def CWD(daily_ts):
  N=daily_ts.shape[0]
  #print(N)
  cwd_max=0
  i=0
  while(True):
    if i>=N:
      break
    else:
      cwd=0
      #print("outer")
      while(True):
        
        if daily_ts[i]>=1 :
          cwd=cwd+1
          #print(cwd,i,daily_ts[i])
          i=i+1
          if i+1>N:
            break;
        else:
          #print(-1,i,daily_ts[i])
          i=i+1
          break
      cwd_max=max(cwd, cwd_max)
      
  return cwd_max

def CWD_annualy(daily_ts,st,en):
  CWD_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    CWD_year[yeaR]=CWD(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return CWD_year

def Rx1day(daily_ts):
  return np.max(daily_ts)

def Rx1day_annualy(daily_ts,st,en):
  Rx1day_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    Rx1day_year[yeaR]=Rx1day(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return Rx1day_year

def Rx1day_annualy_ly(daily_ts,st,en):
  Rx1day_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    Rx1day_year[yeaR]=Rx1day(daily_year_ts)
    days_iter=days_iter+365
  return Rx1day_year

def Rx5day(daily_ts):
  N=np.shape(daily_ts)[0]
  sum_max=0
  for i in range(N-4):
    summ=np.sum(daily_ts[i:i+5])
    sum_max=max(sum_max,summ)

  return sum_max

def Rx5day_annualy(daily_ts,st,en):
  Rx5day_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    Rx5day_year[yeaR]=Rx5day(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return Rx5day_year

def Rx5day_annualy_ly(daily_ts,st,en):
  Rx5day_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    Rx5day_year[yeaR]=Rx5day(daily_year_ts)
    days_iter=days_iter+365
  return Rx5day_year


def SDII(daily_ts):
  sum_RR=np.sum(daily_ts[daily_ts>=1])
  W=np.sum([daily_ts>=1])
  return sum_RR/W


def SDII_annualy(daily_ts,st,en):
  SDII_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    SDII_year[yeaR]=SDII(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return SDII_year


def R10mm(daily_ts):
  return np.sum([daily_ts>=10])

def R10mm_annualy(daily_ts,st,en):
  R10mm_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    R10mm_year[yeaR]=R10mm(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return R10mm_year


def R10mm_annualy_ly(daily_ts,st,en):
  R10mm_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    R10mm_year[yeaR]=R10mm(daily_year_ts)
    days_iter=days_iter+365
  return R10mm_year



def R50mm(daily_ts):
  return np.sum([daily_ts>=50])

def R50mm_annualy(daily_ts,st,en):
  R50mm_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    R50mm_year[yeaR]=R50mm(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return R50mm_year

def R50mm_annualy_ly(daily_ts,st,en):
  R50mm_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    R50mm_year[yeaR]=R50mm(daily_year_ts)
    days_iter=days_iter+365
  return R50mm_year


def R95p(daily_ts):
  N=np.shape(daily_ts)[0]
  data_non_zeros=(daily_ts[daily_ts>=1])

  fita,fitloc,fitscale  = scipy.stats.gamma.fit(data_non_zeros,floc=1)
  cdf2 =  scipy.stats.gamma.cdf(daily_ts, a=fita,loc=fitloc,scale=fitscale)

  R95=scipy.stats.gamma.ppf([0.95], a=fita,loc=fitloc,scale=fitscale)
  #print(R95)
  return np.sum(daily_ts[daily_ts>=R95])

def R95p_annualy(daily_ts,st,en):
  N=np.shape(daily_ts)[0]
  data_non_zeros=(daily_ts[daily_ts>=1])
  fita,fitloc,fitscale  = scipy.stats.gamma.fit(data_non_zeros,floc=1)
  cdf2 =  scipy.stats.gamma.cdf(daily_ts, a=fita,loc=fitloc,scale=fitscale)
  R95=scipy.stats.gamma.ppf([0.95], a=fita,loc=fitloc,scale=fitscale)

  R95p_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):

    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    R95p_year[yeaR]=np.sum(daily_year_ts[daily_year_ts>=R95])
    days_iter=days_iter+uu.isleapyear(st)+365
  return R95p_year



def R95p_annualy_ly(daily_ts,st,en):
  N=np.shape(daily_ts)[0]
  data_non_zeros=(daily_ts[daily_ts>=1])
  fita,fitloc,fitscale  = scipy.stats.gamma.fit(data_non_zeros,floc=1)
  cdf2 =  scipy.stats.gamma.cdf(daily_ts, a=fita,loc=fitloc,scale=fitscale)
  R95=scipy.stats.gamma.ppf([0.95], a=fita,loc=fitloc,scale=fitscale)

  R95p_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):

    daily_year_ts=daily_ts[days_iter:days_iter+365]
    R95p_year[yeaR]=np.sum(daily_year_ts[daily_year_ts>=R95])
    days_iter=days_iter+365
  return R95p_year

def R99p(daily_ts):
  N=np.shape(daily_ts)[0]

  data_non_zeros=(daily_ts[daily_ts>1])
  fita,fitloc,fitscale  = scipy.stats.gamma.fit(data_non_zeros,floc=1)
  cdf2 = scipy.stats.gamma.cdf(daily_ts, a=fita,loc=fitloc,scale=fitscale)

  R99=scipy.stats.gamma.ppf([0.99], a=fita,loc=fitloc,scale=fitscale)
  #print(R99)

  return np.sum(daily_ts[daily_ts>=R99])

def R99p_annualy(daily_ts,st,en):
  N=np.shape(daily_ts)[0]
  data_non_zeros=(daily_ts[daily_ts>1])
  fita,fitloc,fitscale  = scipy.stats.gamma.fit(data_non_zeros,floc=1)
  cdf2 =  scipy.stats.gamma.cdf(daily_ts, a=fita,loc=fitloc,scale=fitscale)
  R99=scipy.stats.gamma.ppf([0.99], a=fita,loc=fitloc,scale=fitscale)

  R99p_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):

    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    R99p_year[yeaR]=np.sum(daily_year_ts[daily_year_ts>=R99])
    days_iter=days_iter+uu.isleapyear(st)+365
  return R99p_year

def R99p_annualy_ly(daily_ts,st,en):
  N=np.shape(daily_ts)[0]
  data_non_zeros=(daily_ts[daily_ts>1])
  fita,fitloc,fitscale  = scipy.stats.gamma.fit(data_non_zeros,floc=1)
  cdf2 =  scipy.stats.gamma.cdf(daily_ts, a=fita,loc=fitloc,scale=fitscale)
  R99=scipy.stats.gamma.ppf([0.99], a=fita,loc=fitloc,scale=fitscale)

  R99p_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):

    daily_year_ts=daily_ts[days_iter:days_iter+365]
    R99p_year[yeaR]=np.sum(daily_year_ts[daily_year_ts>=R99])
    days_iter=days_iter+365
  return R99p_year


def R95pTOT(daily_ts):
  return (100 * R95p(daily_ts) )/ PRCPTOT(daily_ts)


def R95pTOT_annualy(daily_ts,st,en):
  return (100 * np.divide(R95p_annualy(daily_ts,st,en), PRCPTOT_annualy(daily_ts,st,en)))

def R99pTOT(daily_ts):
  return (100 * R99p(daily_ts) )/ PRCPTOT(daily_ts)

def R99pTOT_annualy(daily_ts,st,en):
  return (100 * np.divide(R99p_annualy(daily_ts,st,en), PRCPTOT_annualy(daily_ts,st,en)))

def R99pTOT_annualy_ly(daily_ts,st,en):
  return (100 * np.divide(R99p_annualy_ly(daily_ts,st,en), PRCPTOT_annualy_ly(daily_ts,st,en)))


def PRCPTOT(daily_ts):
  sum_RR=np.sum(daily_ts[daily_ts>=1])
  return sum_RR


def PRCPTOT_annualy(daily_ts,st,en):
  PRCPTOT_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+uu.isleapyear(st)+365]
    PRCPTOT_year[yeaR]=PRCPTOT(daily_year_ts)
    days_iter=days_iter+uu.isleapyear(st)+365
  return PRCPTOT_year

def PRCPTOT_annualy_ly(daily_ts,st,en):
  PRCPTOT_year=np.zeros((en-st+1))
  days_iter=0
  for yeaR in range(en-st+1):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    PRCPTOT_year[yeaR]=PRCPTOT(daily_year_ts)
    days_iter=days_iter+365
  return PRCPTOT_year