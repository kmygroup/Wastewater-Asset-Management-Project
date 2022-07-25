# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:07:35 2022

@author: jpste
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from collections import namedtuple
from sqlalchemy import create_engine


def isNaN(num):
    if float('-inf') < float(num) < float('inf'):
        return False 
    else:
        return True

def contains_word(s, w):
    return (' ' + w + ' ') in (' ' + s + ' ')

def getTown(string):
  End_count=string.find(":")
  if End_count>=0:
    new_String=string[0:End_count]
    return new_String
  else: 
    A= "not found"
    return A

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def WSA(RP_score,FI_score,Model_ID):
    user='root' #user name for the user that is connecting to the database
    password='password' #password for the user that is connecting to the database
    host='localhost' #host name or ip address, for local machines it is localhost otherwise it is the server ip 
    database='test_acua_data' #database which you are connecting to

    engine = create_engine("mysql+pymysql://%s:%s@%s/%s" %(user,
                                                           password,
                                                           host,
                                                           database))

             
    Model_ID_query=""" SELECT 
             *
           FROM
             model_id"""
             
    model_id_tab=pd.read_sql(Model_ID_query,con=engine)
    
    
#    save_loc=Model_ID.get()    
#    db_sel=model_id_tab[model_id_tab['Model_name'].isin([save_loc])]
#    save_sel=str(db_sel.iloc[0,0])
    save_sel=str(Model_ID)
    
    query=""" SELECT 
             *
           FROM
             factors WHERE Model_ID=%s"""%save_sel 
             
             
    New_average=pd.read_sql(query,con=engine)
    New_average['intermediate']=0
    New_average['Weighted Sum Average']=0
    for i in New_average.index:
        score=(RP_score/5)*New_average.loc[i,'RP_Importance']+(FI_score/5)*New_average.loc[i,'FI_Importance']
        score=score/2
        New_average.loc[i,'intermediate']=score
    small_factors=[] 
    for i in New_average.index:
        New_average.loc[i,'Weighted Sum Average']=New_average.loc[i,'intermediate']/New_average['intermediate'].sum()
        if New_average.loc[i,'Weighted Sum Average']== 0.00:
            New_average.drop(i,axis=0) 
        elif New_average.loc[i,'Weighted Sum Average']<= 0.05:
            small_factors.append(i)
            
    
    other_labels=pd.DataFrame(data=New_average.loc[small_factors,"Factor_label"])
# =============================================================================
#     if i in other_labels == 0:
#         df.drop(i)
# =============================================================================
    Large_factors=list(set(New_average.index)-set(small_factors))
    pie_data=New_average.loc[Large_factors,:]
    #print(pie_data)
    pie_data.loc['other']=[0,0,0,0,0,0,0,0,0]

    pie_data.loc['other','Weighted Sum Average']=1-pie_data['Weighted Sum Average'].sum()
    pie_data.loc['other',"Factor_label"]='other'
    #print(pie_data)
    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))
        return "{:.1f}%".format(pct, absolute)
    data=pie_data['Weighted Sum Average'][pie_data['Weighted Sum Average']>0]
    label=pie_data['Factor_label'][pie_data['Weighted Sum Average']>0]
    fig=plt.figure(figsize=(7,4));
    plt.pie(data,
            labels=label,
            autopct=lambda pct: func(pct, data),
            labeldistance=1.15);
    plt.title("Pipeline Factor Assessment");
    plt.close(fig=fig)
    Output=namedtuple("a",['figure','other_labels'])
    return Output(fig,other_labels)