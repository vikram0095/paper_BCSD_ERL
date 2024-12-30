library('MBC')
# Multivariate MBCp bias correction
x1=read.csv("./data_walker/data_oncar.csv",header = FALSE,sep=',')

y1=read.csv("./data_walker/data_ogcm_01.csv",header = FALSE,sep=',')
y2=read.csv("./data_walker/data_ogcm_02.csv",header = FALSE,sep=',')
y3=read.csv("./data_walker/data_ogcm_03.csv",header = FALSE,sep=',')
y4=read.csv("./data_walker/data_ogcm_04.csv",header = FALSE,sep=',')

ncar = as.matrix(x1)
gcm1 = as.matrix(y1)
gcm2 = as.matrix(y2)
gcm3 = as.matrix(y3)
gcm4 = as.matrix(y4)

fit.mbcp <- MBCp(ncar, gcm1, gcm1, ratio.seq=c(0,0,0,0))
gcm_MBCp_01 <- fit.mbcp$mhat.p

fit.mbcp <- MBCp(ncar, gcm2, gcm2, ratio.seq=c(0,0,0,0))
gcm_MBCp_02 <- fit.mbcp$mhat.p
fit.mbcp <- MBCp(ncar, gcm3, gcm3, ratio.seq=c(0,0,0,0))
gcm_MBCp_03 <- fit.mbcp$mhat.p
fit.mbcp <- MBCp(ncar, gcm4, gcm4, ratio.seq=c(0,0,0,0))
gcm_MBCp_04 <- fit.mbcp$mhat.p



write.table(gcm_MBCp_01,"./data_walker/data_MBCp_01.csv",row.names=FALSE, col.names=FALSE, sep=",")
write.table(gcm_MBCp_02,"./data_walker/data_MBCp_02.csv",row.names=FALSE, col.names=FALSE, sep=",")
write.table(gcm_MBCp_03,"./data_walker/data_MBCp_03.csv",row.names=FALSE, col.names=FALSE, sep=",")
write.table(gcm_MBCp_04,"./data_walker/data_MBCp_04.csv",row.names=FALSE, col.names=FALSE, sep=",")


fit.mbcn <- MBCn(ncar, gcm1 ,gcm1, ratio.seq=c(0,0,0,0))
gcm_MBCn_01<- fit.mbcn$mhat.c

fit.mbcn <- MBCn(ncar, gcm2 ,gcm2, ratio.seq=c(0,0,0,0))
gcm_MBCn_02<- fit.mbcn$mhat.c

fit.mbcn <- MBCn(ncar, gcm3 ,gcm3, ratio.seq=c(0,0,0,0))
gcm_MBCn_03<- fit.mbcn$mhat.c

fit.mbcn <- MBCn(ncar, gcm4 ,gcm4, ratio.seq=c(0,0,0,0))
gcm_MBCn_04<- fit.mbcn$mhat.c

write.table(gcm_MBCn_01,"./data_walker/data_MBCn_01.csv",row.names=FALSE, col.names=FALSE, sep=",")
write.table(gcm_MBCn_02,"./data_walker/data_MBCn_02.csv",row.names=FALSE, col.names=FALSE, sep=",")
write.table(gcm_MBCn_03,"./data_walker/data_MBCn_03.csv",row.names=FALSE, col.names=FALSE, sep=",")
write.table(gcm_MBCn_04,"./data_walker/data_MBCn_04.csv",row.names=FALSE, col.names=FALSE, sep=",")

