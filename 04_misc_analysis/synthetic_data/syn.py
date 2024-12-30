import numpy as np
import scipy.stats

def mean_sp(daily_ts):
  
  return daily_ts.mean(axis=0)


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



def Rx1day(daily_ts):
  return np.max(daily_ts)


def Rx5day(daily_ts):
  N=np.shape(daily_ts)[0]
  sum_max=0
  for i in range(N-4):
    summ=np.sum(daily_ts[i:i+5])
    sum_max=max(sum_max,summ)

  return sum_max


def R2mm(daily_ts):
  return np.sum([daily_ts>=2])


def R10mm(daily_ts):
  return np.sum([daily_ts>=10])


def R50mm(daily_ts):
  return np.sum([daily_ts>=50])



def R95p(daily_ts):
  N=np.shape(daily_ts)[0]
  data_non_zeros=(daily_ts[daily_ts>=1])

  fita,fitloc,fitscale  = scipy.stats.gamma.fit(data_non_zeros,floc=1)
  cdf2 =  scipy.stats.gamma.cdf(daily_ts, a=fita,loc=fitloc,scale=fitscale)

  R95=scipy.stats.gamma.ppf([0.95], a=fita,loc=fitloc,scale=fitscale)
  print(R95)
  return np.sum(daily_ts[daily_ts>=R95])


def R99p(daily_ts):
  N=np.shape(daily_ts)[0]

  data_non_zeros=(daily_ts[daily_ts>1])
  fita,fitloc,fitscale  = scipy.stats.gamma.fit(data_non_zeros,floc=1)
  cdf2 = scipy.stats.gamma.cdf(daily_ts, a=fita,loc=fitloc,scale=fitscale)

  R99=scipy.stats.gamma.ppf([0.99], a=fita,loc=fitloc,scale=fitscale)
  #print(R99)

  return np.sum(daily_ts[daily_ts>=R99])
