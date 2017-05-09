#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 01:06:25 2017
d
@author: kevin
"""

# Required Python Machine learning Packages
import pandas as pd

adult_df = pd.read_csv('MLall.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['movieid','userid',
                    'gender','age','occupation','zipcode',
                    'namewords','namepar','year','action',
                    'adventure','animation','childrens',
                    'comedy','crime','documentary','drama',
                    'fantasy','filmnoir','horror','musical',
                    'mystery','romance',
                    'scifi','thriller','war','western','rating']

caracteristicas = ['movieid','namewords','year']





################################################
array_registros_existentes1 = []
print("--- movieid ---")
for registro1 in adult_df['movieid']:
   cambio1 = 0
   for existe1 in array_registros_existentes1:
       if existe1 == registro1:
           cambio1 = 1
           
   if cambio1 == 0:
       array_registros_existentes1.append(registro1)
print(len(array_registros_existentes1))
###############################################
array_registros_existentes2 = []
print("--- namewords ---")
for registro2 in adult_df['namewords']:
   cambio2 = 0
   for existe2 in array_registros_existentes2:
       if existe2 == registro2:
           cambio2 = 1
           
   if cambio2 == 0:
       array_registros_existentes2.append(registro2)

print(len(array_registros_existentes2))
###############################################
array_registros_existentes3 = []
print("--- year ---")
for registro3 in adult_df['year']:
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
dominio_clave_foranea = len(array_registros_existentes1)+len(array_registros_existentes2)+len(array_registros_existentes3)+36
num_muestras_trainning = len(adult_df['movieid'])*0.5
         

print(num_muestras_trainning/dominio_clave_foranea)    