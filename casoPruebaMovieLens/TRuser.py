#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

caracteristicas = ['userid','age','zipcode','occupation']

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
print("--- zipcode ---")
for registro3 in adult_df['zipcode']:
   cambio3 = 0
   for existe3 in array_registros_existentes3:
       if existe3 == registro3:
           cambio3 = 1
           
   if cambio3 == 0:
       array_registros_existentes3.append(registro3)

print(len(array_registros_existentes3))
##################################################
array_registros_existentes4 = []
print("--- occupation ---")
for registro4 in adult_df['occupation']:
   cambio4 = 0
   for existe4 in array_registros_existentes4:
       if existe4 == registro4:
           cambio4 = 1
           
   if cambio4 == 0:
       array_registros_existentes4.append(registro4)

print(len(array_registros_existentes4))
##################################################

print("fin loops \n")

print("TR \n")
dominio_clave_foranea = len(array_registros_existentes1)+len(array_registros_existentes2)+len(array_registros_existentes3)+len(array_registros_existentes4)+2
num_muestras_trainning = len(adult_df['userid'])*0.5
         

print(num_muestras_trainning/dominio_clave_foranea)    