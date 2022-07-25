# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:37:57 2022

@author: jpste
"""
import tkinter as tk
from tkinter import (StringVar)
from tkinter import ttk
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import mysql.connector
from datetime import date
from model_dev import make_model


def pipeline_test_gui():    
    MTgui=tk.Toplevel()
    MTgui.title("Test a pipeline")
    
    
    user='root' #user name for the user that is connecting to the database
    password='password' #password for the user that is connecting to the database
    host='localhost' #host name or ip address, for local machines it is localhost otherwise it is the server ip 
    database='test_acua_data' #database which you are connecting to

    my_conn=mysql.connector.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database)

    engine = create_engine("mysql+pymysql://%s:%s@%s/%s" %(user,
                                                           password,
                                                           host,
                                                           database))
    
    
    
    
    Model_ID_query=""" SELECT 
         *
       FROM
         model_id"""
         
             
    model_id_tab=pd.read_sql(Model_ID_query,con=engine)
    
    #1='model_name'
    saved_model=int(model_id_tab.iloc[0,1])
    #print(saved_model)
# =============================================================================
#     ID_model_list=list(model_id_tab.iloc[1:6,1])
#     model_id_df=model_id_tab.iloc[1:6,1]
#     ID_model_list2=list(model_id_tab.iloc[2:6,1])
# =============================================================================
    
    
    query="""SELECT * From factors WHERE Model_ID=%s""" %saved_model
    fact=pd.read_sql(query,con=engine)

    db_table=["Average_Flow","Mat_Numerical_Category","Pop_Density","Flow_Numerical_Category"]

# =============================================================================
    pd_query=""" SELECT 
         *
     FROM
         pipeline_database
             JOIN
         pop_data ON pop_data.Town_ID = pipeline_database.Town_ID
             JOIN
         flow_category ON flow_category.Flow_Orientation = pipeline_database.Force_or_Gravity_Main
             JOIN
         material_category ON material_category.Material = pipeline_database.Pipe_Type
             JOIN
         pump_stat_avg ON pump_stat_avg.Pump_Sat_ID = pipeline_database.Upstream_Pumpstation_ID"""
    pipe_DB=pd.read_sql(pd_query,con=engine)
    
    pipe_DB.sort_values('Pipe_ID',inplace=True)
    pipe_DB.reset_index(drop=True,inplace=True) 
    
    
    Today=date.today()
    Current_Year=Today.year
    
    pipe_DB['Years_Since_Last_Inspection']=0
    pipe_DB['Remaining_Life']=0
    for i in range(len(pipe_DB['Last_Inspection'])):
        INS_year=pipe_DB.loc[i,'Last_Inspection']
        diff=Current_Year-int(INS_year)
        pipe_DB.loc[i,'Years_Since_Last_Inspection']=diff
        
    for i in range(len(pipe_DB['Replacement_Rehabilitation_Year'])):
        REHY=int(pipe_DB.loc[i,'Replacement_Rehabilitation_Year'])
        if REHY == 0:
            pipe_DB.loc[i,'Remaining_Life']=pipe_DB.loc[i,'Original_Installation']+pipe_DB.loc[i,'Original_Lifespan']-Current_Year
        else:
            pipe_DB.loc[i,'Remaining_Life']=REHY+int(pipe_DB.loc[i,'Replacement_Rehabilitation_Added_Design_Life'])-Current_Year
    
    
    for i in reversed(range(len(pipe_DB['Not_ACUA_Pipeline']))):
        #1 indicates yes it is not an acua pipeline, 0 indicates it is an acua pipeline
        if pipe_DB.loc[i,'Not_ACUA_Pipeline']==1:
            pipe_DB.drop(labels=i,axis=0, inplace=True)
    pipe_DB.reset_index(drop=True,inplace=True)    
    
    for i in reversed(range(len(pipe_DB['Abandoned']))):
        #1 indicates that the pipeline is abandoned, 0 shows it is not
        if pipe_DB.loc[i,'Abandoned']==1:
            pipe_DB.drop(labels=i,axis=0, inplace=True)
            
    pipe_DB.reset_index(drop=True,inplace=True)   
    
    for i in reversed(range(len(pipe_DB['True_Risk_Probability']))):
        if pipe_DB.loc[i,'True_Risk_Probability']==0:
            pipe_DB.drop(labels=i,axis=0, inplace=True)
            
    pipe_DB.reset_index(drop=True,inplace=True)  
# =============================================================================



####################################################   ML Intilization    ######################################################
    Model_ID_query=""" SELECT 
             *
           FROM
             model_id"""
             
    model_id_tab=pd.read_sql(Model_ID_query,con=engine)
    
    #1='model_name'
    saved_model=int(model_id_tab.iloc[0,1])    
    
    factor_query="""SELECT * From factors WHERE Model_ID=%s""" %saved_model
    base_fac=pd.read_sql(factor_query,con=engine)

    RP_fact=base_fac[base_fac['RP_Importance']>0]
    factors=RP_fact["Factors"]
    factor_labels=list(RP_fact["Factor_label"])
    dependentvariable='True_Risk_Probability'
    RP_model=make_model(pipe_DB,factors=factors,dependentvariable=dependentvariable,labels=factor_labels)
    
    
    FI_fact=base_fac[base_fac['FI_Importance']>0]
    factors=FI_fact["Factors"]
    factor_labels=list(FI_fact["Factor_label"])
    dependentvariable='True_Failure_Impact'
    FI_model=make_model(pipe_DB,factors=factors,dependentvariable=dependentvariable,labels=factor_labels)


    names = {}
    
    count=0
    for i,factor in enumerate(fact['Factor_label']):
        count=count+1
        variable_name="field_"+str(i)
        factor_name=fact.loc[i,'Factors']
        if factor=="Average Flowrate":
            label=ttk.Label(MTgui,text='Upstream Pumpstation')
            label.grid(row=i, column=0, padx=5, pady=5)
        elif factor == "Population Density":
            label=ttk.Label(MTgui,text="Town")
            label.grid(row=i, column=0, padx=5, pady=5)
        else:
            label=ttk.Label(MTgui,text=factor)
            label.grid(row=i, column=0, padx=5, pady=5)

        names[variable_name]=(StringVar(),factor,factor_name)

        if fact.loc[i,"Factors"] in db_table:
            what_table=fact.loc[i,"Factors"]
            if what_table=="Average_Flow":
                query='SELECT Pump_Station FROM pump_stat_avg'
                column='Pump_Station'

            elif what_table=="Mat_Numerical_Category":
                query='SELECT Material FROM material_category'
                column='Material'

            elif what_table=="Flow_Numerical_Category":
                query='SELECT Flow_Orientation FROM flow_category'
                column='Flow_Orientation' 

            elif what_table=="Pop_Density":
                query='SELECT Location FROM pop_data'
                column='Location'

            dropvalues= pd.read_sql(query,con=engine)
            opt=list(dropvalues[column])
            PSSearch=ttk.OptionMenu(MTgui, names[variable_name][0], opt[0],*opt)
            PSSearch.config(width=15)
            PSSearch.grid(row=i, column=1, padx=5, pady=5)
        else:

            entry = ttk.Entry(MTgui,textvariable=names[variable_name][0])
            entry.grid(row=i, column=1, padx=5, pady=5)

    RP_label=ttk.Label(MTgui,text="Risk Probability Score")
    RP_label.grid(row=count+2, column=0, padx=5, pady=5)
    RP_Eval=StringVar()
    RP_entry = ttk.Entry(MTgui,textvariable=RP_Eval)
    RP_entry.grid(row=count+2, column=1, padx=5, pady=5)

    FI_label=ttk.Label(MTgui,text="Failure Impact Score")
    FI_label.grid(row=count+3, column=0, padx=5, pady=5)
    FI_Eval=StringVar()
    FI_entry = ttk.Entry(MTgui,textvariable=FI_Eval)
    FI_entry.grid(row=count+3, column=1, padx=5, pady=5)

    def find_values():
        try:
            
            Model_ID_query2=""" SELECT * FROM model_id"""
            model_id_tab2=pd.read_sql(Model_ID_query2,con=engine)
            #1='model_name'
            saved_model2=int(model_id_tab2.iloc[0,1])

            
            FI_query="""SELECT Factors, Factor_label From factors Where FI_Importance >0 and Model_ID=%s""" %saved_model2
            FI_fact=pd.read_sql(FI_query,con=engine)
            FI_columns=list(FI_fact['Factors'])
          
            FI_data=pd.DataFrame(columns=FI_columns)

            RP_query="""SELECT Factors, Factor_label From factors Where RP_Importance >0 and Model_ID=%s""" %saved_model2
            RP_fact=pd.read_sql(RP_query,con=engine)
            RP_columns=list(RP_fact['Factors'])
            
            RP_data=pd.DataFrame(columns=RP_columns)

            for i,var in enumerate(names):
                if names[var][2] in RP_columns:
                    if names[var][2] in db_table:
                        if names[var][2] == "Average_Flow":
                            query='SELECT Average_Flow FROM pump_stat_avg WHERE Pump_Station = "%s"'%names[var][0].get()
                        elif names[var][2] == "Mat_Numerical_Category":
                            query='SELECT Mat_Numerical_Category FROM material_category WHERE Material = "%s"'%names[var][0].get()
                        elif names[var][2] == "Pop_Density":
                            query='SELECT Pop_Density FROM pop_data WHERE Location = "%s"'%names[var][0].get()
                        elif names[var][2] == "Flow_Numerical_Category":
                            query='SELECT FLow_Numerical_category FROM flow_category WHERE Flow_Orientation = "%s"'%names[var][0].get()
                        RP_value=pd.read_sql(query,con=engine)
                        RP_pipe_val=RP_value.iloc[0,0]
                    else:
                        RP_pipe_val=int(names[var][0].get())
                    RP_data.loc[0,names[var][2]]=RP_pipe_val
            for i,var in enumerate(names):
                if names[var][2] in FI_columns:
                    if names[var][2] in db_table:
                        if names[var][2] == "Average_Flow":
                            query='SELECT Average_Flow FROM pump_stat_avg WHERE Pump_Station = "%s"'%names[var][0].get()
                        elif names[var][2] == "Mat_Numerical_Category":
                            query='SELECT Mat_Numerical_Category FROM material_category WHERE Material = "%s"'%names[var][0].get()
                        elif names[var][2] == "Pop_Density":
                            query='SELECT Pop_Density FROM pop_data WHERE Location = "%s"'%names[var][0].get()
                        elif names[var][2] == "Flow_Numerical_Category":
                            query='SELECT FLow_Numerical_category FROM flow_category WHERE Flow_Orientation = "%s"'%names[var][0].get()
                        FI_value=pd.read_sql(query,con=engine)
                        FI_pipe_val=FI_value.iloc[0,0]
                    else:
                        FI_pipe_val=int(names[var][0].get())
                    FI_data.loc[0,names[var][2]]=FI_pipe_val
                    
            RP_Score=RP_model.model.predict(RP_data)
            FI_Score=FI_model.model.predict(FI_data)
            RP_Eval.set(str(RP_Score[0]))
            FI_Eval.set(str(FI_Score[0]))
        except Exception:
            Errorgui = tk.Tk()
            Errorgui.title("Error!")
            w = 650 # width for the Tk root
            h = 100 # height for the Tk root
            ws = Errorgui.winfo_screenwidth() # width of the screen
            hs = Errorgui.winfo_screenheight() # height of the screen
            x = (ws/2) - (w/2)
            y = (hs/2) - (h/2)
            Errorgui.geometry('%dx%d+%d+%d' % (w, h, x, y-25))     
            label=ttk.Label(Errorgui,
                            text="Helpful Hints: \nOnly numbers are allowed in text boxes!\nMake sure everything is filled out!\nIf you changed the base model in the other application try closing the predictor tool and re-opening it again!")
            label.pack(pady=10,padx=10,anchor='center')   


    Execute_Button=ttk.Button(MTgui,text='Find Values',width=25,command=find_values)
    Execute_Button.grid(row=count+1, column=0, padx=0, pady=0,sticky='ew',columnspan=2)

    MTgui.mainloop()