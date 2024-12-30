import numpy as np
import scipy.stats


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
        
        if daily_ts[i]<=1 :
          cdd=cdd+1
          i=i+1
          if i+1>N:
            break;
        else:
          #print(-1,i,daily_ts[i])
          i=i+1
          break
      cdd_max=max(cdd, cdd_max)
      
  return cdd_max


def CDD_annualy(daily_ts,Nyears):
  CDD_year=np.zeros((Nyears))
  days_iter=0
  for yeaR in range(Nyears):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    CDD_year[yeaR]=CDD(daily_year_ts)
    days_iter=days_iter+365
  return CDD_year





def sum_annualy(daily_ts,Nyears):
  Rx1day_year=np.zeros((Nyears))
  days_iter=0
  for yeaR in range(Nyears):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    Rx1day_year[yeaR]=np.sum(daily_year_ts)
    days_iter=days_iter+365
  return Rx1day_year


def mean_annualy(daily_ts,Nyears):
  Rx1day_year=np.zeros((Nyears))
  days_iter=0
  for yeaR in range(Nyears):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    Rx1day_year[yeaR]=np.sum(daily_year_ts)
    days_iter=days_iter+365
  return Rx1day_year





def Rx1day(daily_ts):
  return np.max(daily_ts)

def Rx1day_annualy(daily_ts,Nyears):
  Rx1day_year=np.zeros((Nyears))
  days_iter=0
  for yeaR in range(Nyears):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    Rx1day_year[yeaR]=Rx1day(daily_year_ts)
    days_iter=days_iter+365
  return Rx1day_year

# def Rx5day(daily_ts):
#   N=np.shape(daily_ts)[0]
#   sum_max=0
#   for i in range(N-4):
#     summ=np.sum(daily_ts[i:i+5])
#     sum_max=max(sum_max,summ)
#   return sum_max


# def Rx5day_annualy(daily_ts,Nyears):
#   Rx5day_year=np.zeros((Nyears))
#   days_iter=0
#   for yeaR in range(Nyears):
#     daily_year_ts=daily_ts[days_iter:days_iter+365]
#     Rx5day_year[yeaR]=Rx5day(daily_year_ts)
#     days_iter=days_iter+365
#   return Rx5day_year


def R20mm(daily_ts):
  return np.sum([daily_ts>=20])

def R20mm_annualy(daily_ts,Nyears):
  R20mm_year=np.zeros((Nyears))
  days_iter=0
  for yeaR in range(Nyears):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    R20mm_year[yeaR]=R20mm(daily_year_ts)
    days_iter=days_iter+365
  return R20mm_year


def R50mm(daily_ts):
  return np.sum([daily_ts>=50])

def R50mm_annualy(daily_ts,Nyears):
  R50mm_year=np.zeros((Nyears))
  days_iter=0
  for yeaR in range(Nyears):
    daily_year_ts=daily_ts[days_iter:days_iter+365]
    R50mm_year[yeaR]=R50mm(daily_year_ts)
    days_iter=days_iter+365
  return R50mm_year


# def R95p(daily_ts):
#     data_non_zeros=(daily_ts[daily_ts>=1])
#     p95 = np.percentile(data_non_zeros,95)
#     return np.sum(data_non_zeros[data_non_zeros>=p95])


# def R95p_annualy(daily_ts,Nyears):
#     R95p_year=np.zeros((Nyears))
#     days_iter=0
#     for yeaR in range(Nyears):
#         daily_year_ts=daily_ts[days_iter:days_iter+365]
#         R95p_year[yeaR]=np.sum(daily_year_ts[daily_year_ts>=R95])
#         days_iter=days_iter+365
#     return R95p_year



def R99p(daily_ts):
    data_non_zeros=(daily_ts[daily_ts>=1])
    # print(data_non_zeros.shape[0]/daily_ts.shape[0] *100)
    if data_non_zeros.shape[0]<1:
        p99 = 0
    else:
        p99 = np.percentile(data_non_zeros,99)
    return np.sum(data_non_zeros[data_non_zeros>=p99])


def R99p_annualy(daily_ts,Nyears):
    R99p_year=np.zeros((Nyears))
    days_iter=0
    for yeaR in range(Nyears):
        daily_year_ts=daily_ts[days_iter:days_iter+365]
        R99p_year[yeaR]=R99p(daily_year_ts)#np.sum(daily_year_ts[daily_year_ts>=R99])
        days_iter=days_iter+365
    return R99p_year




# def R95pTOT(daily_ts):
#   return (100 * R95p(daily_ts) )/ PRCPTOT(daily_ts)


# def R95pTOT_annualy(daily_ts,Nyears):
#   return (100 * np.divide(R95p_annualy(daily_ts,Nyears), PRCPTOT_annualy(daily_ts,Nyears)))

# def R99pTOT(daily_ts):
#   return (100 * R99p(daily_ts) )/ PRCPTOT(daily_ts)

# def R99pTOT_annualy(daily_ts,Nyears):
#   return (100 * np.divide(R99p_annualy(daily_ts,Nyears), PRCPTOT_annualy(daily_ts,Nyears)))

# def PRCPTOT(daily_ts):
#   sum_RR=np.sum(daily_ts[daily_ts>=1])
#   return sum_RR

# def PRCPTOT_annualy(daily_ts,Nyears):
#   PRCPTOT_year=np.zeros((Nyears))
#   days_iter=0
#   for yeaR in range(Nyears):
#     daily_year_ts=daily_ts[days_iter:days_iter+365]
#     PRCPTOT_year[yeaR]=PRCPTOT(daily_year_ts)
#     days_iter=days_iter+365
#   return PRCPTOT_year
