#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 02:06:18 2017

@author: kevin
"""

# Required Python Machine learning Packages
import pandas as pd


adult_df = pd.read_csv('BCall.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['userid','bookid',
					'titlewords','authorwords',
					'year','publisher','country','age','rating']

caracteristicas = ['userid','age','country']

################################################
array_registros_existentes1 = []
print("--- userid ---")
for registro1 in adult_df['userid']:
   cambio1 = 0
   for existe1 in array_registros_existentes1:
       if existe1 == registro1:
           cambio1 = 1
           
   if cambio1 == 0:
       array_registros_existentes1.append(registro1)
print(len(array_registros_existentes1))
###############################################
array_registros_existentes2 = []
print("--- age ---")
for registro2 in adult_df['age']:
   cambio2 = 0
   for existe2 in array_registros_existentes2:
       if existe2 == registro2:
           cambio2 = 1
           
   if cambio2 == 0:
       array_registros_existentes2.append(registro2)

print(len(array_registros_existentes2))
###############################################
array_registros_existentes3 = []
print("--- country ---")
for registro3 in adult_df['country']:
   cambio3 = 0
   for existe3 in array_registros_existentes3:
       if existe3 == registro3:
           cambio3 = 1
           
   if cambio3 == 0:
       array_registros_existentes3.append(registro3)

print(len(array_registros_existentes3))
##################################################

print("fin loops \n")

print("TR \n")
dominio_clave_foranea = len(array_registros_existentes1)+len(array_registros_existentes2)+len(array_registros_existentes3)
num_muestras_trainning = len(adult_df['userid'])*0.5
         

print(num_muestras_trainning/dominio_clave_foranea)    