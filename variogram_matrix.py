# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:58:10 2021
"""
import numpy as np
import pandas as pd
import copy
import math

def variogram_matrix(grd, R = 5, dx = 1, dy = 1, zero_out = False):
    dat = grd.copy()
    out = {}
    '''
    if (zero.out) 
        dat[dat == 0] <- NA
    N <- ncol(dat)
    M <- nrow(dat)
    m <- min(c(round(R/dx), M))    #round函数表示返回小数点后几位
    n <- min(c(round(R/dy), N))
    ind <- rbind(as.matrix(expand.grid(0, 1:n)), 
                 as.matrix(expand.grid(1:m, 0)), 
                 as.matrix(expand.grid(c(-(m:1), 1:m), 1:n)))
    '''
    if (zero_out):
        dat[dat == 0] = np.nan
    N = np.shape(dat)[1]
    M = np.shape(dat)[0]
    m = min(round(R/dx), M)    #round函数表示返回小数点后几位
    n = min(round(R/dy), N)
    #ind变量拼凑
    ind1 = np.vstack((np.zeros(n), np.arange(1, n+1))).T
    ind2 = np.vstack((np.arange(1, m+1), np.zeros(m))).T
    ind3 = np.tile(np.hstack((np.arange(-m, 0), np.arange(1, m+1))).T, n)
    ind4 = np.vstack((ind3, np.repeat(np.arange(1, n+1), len(ind3)/n))).T                     
    ind = np.vstack((ind1, ind2, ind4))
    
    '''
    d <- sqrt((dx * ind[, 1])^2 + (dy * ind[, 2])^2)
    good <- (d > 0) & (d <= R)
    ind <- ind[good, ]
    d <- d[good]
    ind <- ind[order(d), ]
    d <- sort(d)
    '''
    d = np.sqrt((dx * ind[:, 0])**2 + (dy * ind[:, 1])**2)
    good = (d > 0) & (d <= R)
    ind = ind[good, :]
    d = d[good]
    ind = ind[np.argsort(d), :]
    d = np.sort(d)
    '''
    nbin <- nrow(ind)
    holdVG <- rep(NA, nbin)
    holdN <- rep(NA, nbin)
    '''
    nbin = len(ind)
    holdVG = np.repeat(np.nan, nbin)
    holdN = np.repeat(np.nan, nbin)    
    #自定义SI函数
    '''
    SI <- function(ntemp, delta) {
        n1 <- 1:ntemp
        n2 <- n1 + delta
        good <- (n2 >= 1) & (n2 <= ntemp)
        cbind(n1[good], n2[good])
    }
    '''
    def SI(ntemp, delta):
        n1 = np.arange(1, ntemp+1)
        n2 = n1 + delta
        good = (n2 >= 1) & (n2 <= ntemp)
        return np.vstack((n1[good], n2[good])).T
    
    '''
    for (k in 1:nbin) {
        MM <- SI(M, ind[k, 1])
        NN <- SI(N, ind[k, 2])
        numNA <- sum(is.na(dat[MM[, 1], NN[, 1]]) | 
                     is.na(dat[MM[, 2], NN[, 2]]), na.rm = TRUE)                                                                 
        holdN[k] <- length(MM) * length(NN) - numNA
        BigDiff <- (dat[MM[, 1], NN[, 1]] - dat[MM[, 2], NN[, 2]])
        holdVG[k] <- mean(0.5 * (BigDiff)^2, na.rm = TRUE)
    }
    '''
    for k in range(nbin):
        #print(k)
        MM = SI(ntemp=M, delta=ind[k, 0])
        NN = SI(ntemp=N, delta=ind[k, 1])
        
        i0 = MM[:, 0].astype(int)
        j0 = NN[:, 0].astype(int)
        i, j = np.meshgrid(i0, j0)
        dat_1 = dat[i-1, j-1].T
        ii0 = MM[:, 1].astype(int)
        jj0 = NN[:, 1].astype(int)
        ii, jj = np.meshgrid(ii0, jj0)
        dat_2 = dat[ii-1, jj-1].T        
        numNA = sum(sum((dat_1 == np.nan) | (dat_2 == np.nan)))
        holdN[k] = np.size(MM) * np.size(NN) - numNA       
        BigDiff = dat_1 - dat_2
        holdVG[k] = np.mean(0.5 * BigDiff**2)
        
    '''
    top <- tapply(holdVG * holdN, d, FUN = "sum")    #tapply(向量数据，分组标志，运算函数),分组计算
    bottom <- tapply(holdN, d, FUN = "sum")
    dcollapsed <- as.numeric(names(bottom))
    vgram <- top/bottom
    '''
    #转为pd.DataFrame再分组计算
    top_df = pd.DataFrame({'key':d, 'data':holdVG * holdN})
    top_grouped = top_df['data'].groupby(top_df['key'])    #分组计算求和
    top = top_grouped.sum()
    
    bottom_df = pd.DataFrame({'key':d, 'data':holdN})
    bottom_grouped = bottom_df['data'].groupby(bottom_df['key'])
    bottom = bottom_grouped.sum()
    #dcollapsed <- as.numeric(names(bottom))
    dcollapsed = np.unique(d)
    vgram = top/bottom     #求权重平均值
    '''
    dimnames(vgram) <- NULL
    out <- list(vgram = vgram, d = dcollapsed, ind = ind, d.full = d, 
        vgram.full = holdVG, N = holdN, dx = dx, dy = dy)
    class(out) <- "vgram.matrix"
    '''
    #dimnames(vgram) <- NULL
    out = {"vgram":vgram, "d":dcollapsed, "ind":ind, "d_full":d, 
           "vgram_full":holdVG, "N":holdN, "dx":dx, "dy":dy, "class":"vgram.matrix"}
   
    return out 

if __name__ == "__main__":
    dat = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\pert000.csv")
    look_VGM = variogram_matrix(grd = dat.values)    #读入的是dataframe数据，有行列名称





















