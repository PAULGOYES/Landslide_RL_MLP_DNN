#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:44:12 2020

@author: Paul Goyes-Peñafiel, Alejandra Hernandez-Rojas (2020)
Este script permite calcular el PCA usando como valores de entrada varios RASTER, y eligiendo un número de componentes determinado.
"""

import gdal
import numpy as np 

rasternames = ["Flow_Accumulation","Flow_Length",
               "General_Curvature","Plan_Curvature","Profile_Curvature",
               "Slope","TRI","TWI"]

fn = "RASTER_TIFF"



v1 = gdal.Open(fn + "/"+rasternames[0]+".tif")

col   = v1.RasterXSize
rows  = v1.RasterYSize
nelem = col*rows
driver = v1.GetDriver()

v2 = gdal.Open(fn + "/"+rasternames[1]+".tif")
v3 = gdal.Open(fn + "/"+rasternames[2]+".tif")
v4 = gdal.Open(fn + "/"+rasternames[3]+".tif")
v5 = gdal.Open(fn + "/"+rasternames[4]+".tif")
v6 = gdal.Open(fn + "/"+rasternames[5]+".tif")
v7 = gdal.Open(fn + "/"+rasternames[6]+".tif")
v8 = gdal.Open(fn + "/"+rasternames[7]+".tif")

v1val = v1.GetRasterBand(1).ReadAsArray().flatten()
v2val = v2.GetRasterBand(1).ReadAsArray().flatten()
v3val = v3.GetRasterBand(1).ReadAsArray().flatten()
v4val = v4.GetRasterBand(1).ReadAsArray().flatten()
v5val = v5.GetRasterBand(1).ReadAsArray().flatten()
v6val = v6.GetRasterBand(1).ReadAsArray().flatten()
v7val = v7.GetRasterBand(1).ReadAsArray().flatten()
v8val = v8.GetRasterBand(1).ReadAsArray().flatten()

DATA = np.stack((v1val,v2val,v3val,v4val,v5val,v6val,v7val,v8val),axis=1)

NanValues = np.where(v1val == -99999)[0]
cP        = np.arange(0,nelem)
cPP       = np.delete(cP, NanValues, axis=0)



X = np.delete(DATA, NanValues, axis=0)

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)



#################################################
# Write PCA data in the raster file
#################################################


#pca1 = gdal.Open(fn + "/PCA_RASTER/" + "PCA1" + ".tif")

pca1 = driver.Create(fn + "/PCA_RASTER/" + "PCA11" + ".tif", col, rows, 1, gdal.GDT_Float32)

# Write metadata
pca1.SetGeoTransform(v1.GetGeoTransform())
pca1.SetProjection(v1.GetProjection())

pca1dataarray = np.zeros((rows,col)).flatten()


for i in range(cPP.shape[0]):
    pca1dataarray[cPP[i]]=Y_sklearn[i,0]
    
for i in range(NanValues.shape[0]):
    pca1dataarray[NanValues[i]]=-99999


pca1.GetRasterBand(1).WriteArray(pca1dataarray.reshape(rows,col))
pca1.GetRasterBand(1).SetNoDataValue(-99999)

  
pca1 = None
del pca1

#
pca2 = driver.Create(fn + "/PCA_RASTER/" + "PCA22" + ".tif", col, rows, 1, gdal.GDT_Float32)
pca2.SetGeoTransform(v1.GetGeoTransform())
pca2.SetProjection(v1.GetProjection())

pca2dataarray = np.zeros((rows,col)).flatten()

for i in range(cPP.shape[0]):
    pca2dataarray[cPP[i]]=Y_sklearn[i,1]
    
for i in range(NanValues.shape[0]):
    pca2dataarray[NanValues[i]]=-99999


pca2.GetRasterBand(1).WriteArray(pca2dataarray.reshape(rows,col))
pca2.GetRasterBand(1).SetNoDataValue(-99999)

  
pca2 = None
del pca2


