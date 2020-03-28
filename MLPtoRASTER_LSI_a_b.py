#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:40:17 2020

@author: Paul Goyes-Peñafiel, Alejandra Hernandez-Rojas. (2020)
Con este script de python se realiza la aplicacion de los modelos
de redes neuronales sobre los raster
"""

import gdal
import numpy as np 
import matplotlib.pyplot as plt
rasternames = ["K1K2K3K4K5.tif"]

#rasternames=["PC1_WoE.tif","PC2_WoE.tif"]
v1 = gdal.Open(rasternames[0])

col   = v1.RasterXSize
rows  = v1.RasterYSize
nelem = col*rows
driver = v1.GetDriver()


v1val = v1.GetRasterBand(1).ReadAsArray().flatten()
v2val = v1.GetRasterBand(2).ReadAsArray().flatten()
v3val = v1.GetRasterBand(3).ReadAsArray().flatten()
v4val = v1.GetRasterBand(4).ReadAsArray().flatten()
v5val = v1.GetRasterBand(5).ReadAsArray().flatten()
#v1.GetRasterBand(4).GetNoDataValue()
v1val[v1val==-9999] = None
v2val[v2val==-9999] = None
v3val[v3val==0] = None
v4val[v4val==0] = None
v5val[v5val==-0] = None
#plt.imshow(v5val)

#v1.GetRasterBand(1).GetNoDataValue()

DATA = np.stack((v1val.flatten(),v2val.flatten(),v3val.flatten(),v4val.flatten(),v5val.flatten()),axis=1)

#cargar modelo entrenado desde TENSORFLOW
result = model.predict(DATA)

plt.hist(result,bins=30,histtype='bar', ec='black',color='b')

plt.imshow(result.reshape((rows,col)),cmap='jet'), plt.colorbar()

#write_result

pca1 = driver.Create("DNN_LSI_OP_B" + ".tif", col, rows, 1, gdal.GDT_Float32)

# Write metadata
pca1.SetGeoTransform(v1.GetGeoTransform())
pca1.SetProjection(v1.GetProjection())

pca1dataarray = result.copy()
pca1dataarray[pca1dataarray==None]=-9999


pca1.GetRasterBand(1).WriteArray(pca1dataarray.reshape(rows,col))
pca1.GetRasterBand(1).SetNoDataValue(-9999)

  
pca1 = None
del pca1
