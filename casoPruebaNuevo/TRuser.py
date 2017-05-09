#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 03:16:50 2017

@author: kevin
"""

# Required Python Machine learning Packages
import pandas as pd

adult_df = pd.read_csv('adult.data',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['userid','educationid','age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

caracteristicas = ['userid','age','sex','marital_status','relationship','race','occupation',
                   'hours_per_week','native_country']





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
print("--- sex ---")
for registro2 in adult_df['sex']:
   cambio2 = 0
   for existe2 in array_registros_existentes2:
       if existe2 == registro2:
           cambio2 = 1
           
   if cambio2 == 0:
       array_registros_existentes2.append(registro2)

print(len(array_registros_existentes2))
###############################################
array_registros_existentes3 = []
print("--- age ---")
for registro3 in adult_df['age']:
   cambio3 = 0
   for existe3 in array_registros_existentes3:
       if existe3 == registro3:
           cambio3 = 1
           
   if cambio3 == 0:
       array_registros_existentes3.append(registro3)

print(len(array_registros_existentes3))
##################################################
array_registros_existentes4 = []
print("--- marital_status ---")
for registro4 in adult_df['marital_status']:
   cambio4 = 0
   for existe4 in array_registros_existentes4:
       if existe4 == registro4:
           cambio4 = 1
           
   if cambio4 == 0:
       array_registros_existentes4.append(registro4)

print(len(array_registros_existentes4))
##################################################

array_registros_existentes5 = []
print("--- relationship ---")
for registro5 in adult_df['relationship']:
   cambio5 = 0
   for existe5 in array_registros_existentes5:
       if existe5 == registro5:
           cambio5 = 1
           
   if cambio5 == 0:
       array_registros_existentes5.append(registro5)

print(len(array_registros_existentes5))
##################################################

array_registros_existentes6 = []
print("--- race ---")
for registro6 in adult_df['race']:
   cambio6 = 0
   for existe6 in array_registros_existentes6:
       if existe6 == registro6:
           cambio6 = 1
           
   if cambio6 == 0:
       array_registros_existentes6.append(registro6)

print(len(array_registros_existentes6))
##################################################

array_registros_existentes7 = []
print("--- occupation ---")
for registro7 in adult_df['occupation']:
   cambio7 = 0
   for existe7 in array_registros_existentes7:
       if existe7 == registro7:
           cambio7 = 1
           
   if cambio7 == 0:
       array_registros_existentes7.append(registro7)

print(len(array_registros_existentes7))
##################################################

array_registros_existentes8 = []
print("--- hours_per_week ---")
for registro8 in adult_df['hours_per_week']:
   cambio8 = 0
   for existe8 in array_registros_existentes8:
       if existe8 == registro8:
           cambio8 = 1
           
   if cambio8 == 0:
       array_registros_existentes8.append(registro8)

print(len(array_registros_existentes8))
##################################################

array_registros_existentes9 = []
print("--- native_country ---")
for registro9 in adult_df['native_country']:
   cambio9 = 0
   for existe9 in array_registros_existentes9:
       if existe9 == registro9:
           cambio9 = 1
           
   if cambio9 == 0:
       array_registros_existentes9.append(registro9)

print(len(array_registros_existentes9))
##################################################
print("fin loops \n")

print("TR \n")
dominio_clave_foranea = len(array_registros_existentes1)+len(array_registros_existentes2)+len(array_registros_existentes3)+len(array_registros_existentes4)+len(array_registros_existentes5)+len(array_registros_existentes6)+len(array_registros_existentes7)+len(array_registros_existentes8)+len(array_registros_existentes9)
                        
num_muestras_trainning = len(adult_df['userid'])*0.5
         

print(num_muestras_trainning/dominio_clave_foranea)    