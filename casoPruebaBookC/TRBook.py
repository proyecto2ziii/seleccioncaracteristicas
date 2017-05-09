#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 02:19:27 2017

@author: kevin
"""

# Required Python Machine learning Packages
import pandas as pd


adult_df = pd.read_csv('BCall.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['userid','bookid',
					'titlewords','authorwords',
					'year','publisher','country','age','rating']

caracteristicas = ['bookid','year','publisher','titlewords','authorwords']

################################################
array_registros_existentes1 = []
print("--- bookid ---")
for registro1 in adult_df['bookid']:
   cambio1 = 0
   for existe1 in array_registros_existentes1:
       if existe1 == registro1:
           cambio1 = 1
           
   if cambio1 == 0:
       array_registros_existentes1.append(registro1)
print(len(array_registros_existentes1))
###############################################
array_registros_existentes2 = []
print("--- year ---")
for registro2 in adult_df['year']:
   cambio2 = 0
   for existe2 in array_registros_existentes2:
       if existe2 == registro2:
           cambio2 = 1
           
   if cambio2 == 0:
       array_registros_existentes2.append(registro2)

print(len(array_registros_existentes2))
###############################################
array_registros_existentes3 = []
print("--- publisher ---")
for registro3 in adult_df['publisher']:
   cambio3 = 0
   for existe3 in array_registros_existentes3:
       if existe3 == registro3:
           cambio3 = 1
           
   if cambio3 == 0:
       array_registros_existentes3.append(registro3)

print(len(array_registros_existentes3))
##################################################
array_registros_existentes4 = []
print("--- titlewords ---")
for registro4 in adult_df['titlewords']:
   cambio4 = 0
   for existe4 in array_registros_existentes4:
       if existe4 == registro4:
           cambio4 = 1
           
   if cambio4 == 0:
       array_registros_existentes4.append(registro4)

print(len(array_registros_existentes4))
##################################################
array_registros_existentes5 = []
print("--- authorwords ---")
for registro5 in adult_df['authorwords']:
   cambio5 = 0
   for existe5 in array_registros_existentes5:
       if existe5 == registro5:
           cambio5 = 1
           
   if cambio5 == 0:
       array_registros_existentes5.append(registro5)

print(len(array_registros_existentes5))
##################################################

print("fin loops \n")

print("TR \n")
dominio_clave_foranea = len(array_registros_existentes1)+len(array_registros_existentes2)+len(array_registros_existentes3)+len(array_registros_existentes4)+len(array_registros_existentes5)
num_muestras_trainning = len(adult_df['bookid'])*0.5
         

print(num_muestras_trainning/dominio_clave_foranea)    