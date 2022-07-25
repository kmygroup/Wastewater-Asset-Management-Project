# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:58:21 2022

@author: jpste
"""

#Notes to self

#Graphs --> Tools--> Preferences--> IPython console --> Graphics --> Graphics backend --> Make a selection --> Restart kernel

import tkinter as tk
from tkinter import (StringVar)
from tkinter import ttk
import numpy as np
import pandas as pd
import time
from datetime import date
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import os
from sqlalchemy import create_engine
import mysql.connector

from misc_functions import (isNaN,contains_word,getTown,is_float,WSA)
from model_dev import make_model

from Excel_func import make_report

from GUI_func import (insert_var,delete_record)
from pipe_test_GUI import (pipeline_test_gui)



# ###########################################GUI Formation and Settings###################################################
#Make the parent window
ACUAgui = tk.Tk()
ACUAgui.title("ACUA Pipeline Asset Summary")

w = 1500 # width for the Tk root
h = 800 # height for the Tk root
# get screen width and height
ws = ACUAgui.winfo_screenwidth() # width of the screen
hs = ACUAgui.winfo_screenheight() # height of the screen
# calculate x and y coordinates for the Tk root window
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
#set the dimensions of the screen  and where it is placed
ACUAgui.geometry('%dx%d+%d+%d' % (w, h, x, y-25))

###################################   Forming Database Connection  ######################################################

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

mycursor=my_conn.cursor()

###################################   Initalization of Model choice  #################################################################
Model_ID_query=""" SELECT 
         *
       FROM
         model_id"""
         
         
model_id_tab=pd.read_sql(Model_ID_query,con=engine)

#1='model_name'
saved_model=int(model_id_tab.iloc[0,1])
ID_model_list=list(model_id_tab.iloc[1:6,1])
model_id_df=model_id_tab.iloc[1:6,1]
ID_model_list2=list(model_id_tab.iloc[2:6,1])
##variables used in selecting the model

Model_ID= StringVar()
Save_ID= StringVar()



###################################   Intialization of Dataframe from Database  ######################################################

def make_pipe_db():
    global pipe_DB,Pipe_DB_All,RP_model,FI_model,base_fac
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
    
    #pipe_DB.drop(columns=["RF_Risk_Probability","RF_Failure_Impact","OPMN",'Years_Since_Last_Inspection','Remaining_Life'],inplace=True)
    
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
    
    Pipe_DB_All=pipe_DB.copy()
    
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
    
    
    Numeric_col=pipe_DB.select_dtypes(exclude=['object'])
    
    ####################################################   ML Intilization    ######################################################
    factor_query="""SELECT * From factors WHERE Model_ID=%s""" %saved_model
    base_fac=pd.read_sql(factor_query,con=engine)
    
    RP_fact=base_fac[base_fac['RP_Importance']>0]
    factors=RP_fact["Factors"]
    factor_labels=list(RP_fact["Factor_label"])
    dependentvariable='True_Risk_Probability'
    RP_model=make_model(pipe_DB,factors=factors,dependentvariable=dependentvariable,labels=factor_labels)
    all_RP_data=Pipe_DB_All.loc[:,factors]
    RP_prediction=RP_model.model.predict(all_RP_data)
    
    
    
    FI_fact=base_fac[base_fac['FI_Importance']>0]
    fi_factors=FI_fact["Factors"]
    fi_factor_labels=list(FI_fact["Factor_label"])
    dependentvariable='True_Failure_Impact'
    FI_model=make_model(pipe_DB,factors=fi_factors,dependentvariable=dependentvariable,labels=fi_factor_labels)
    all_FI_data=Pipe_DB_All.loc[:,fi_factors]
    FI_prediction=FI_model.model.predict(all_FI_data)
    
    OPMN_vals=RP_prediction*FI_prediction
    
    
    Pipe_DB_All['Random Forest RP']=RP_prediction
    Pipe_DB_All['Random Forest FI']=FI_prediction
    Pipe_DB_All['Random Forest OPMN']=OPMN_vals

    return pipe_DB,Pipe_DB_All,RP_model,FI_model,base_fac

make_pipe_db()

####################################################GUI Style Guide######################################################
Style=ttk.Style()
#Styles all the tabs
Style.configure('BLCK.TFrame',background= 'black')
Style.configure('BLU.TFrame',background= 'light blue')
Style.configure('BLU.TLabel',background='light blue')
Style.configure('TLabelFrame',background='light blue')
Style.configure('Error.TLabel',foreground='red')
Style.configure('Head.TLabel',font=(('Arial',12)))

Style.configure('BW.Treeview')
Style.map('BW.Treeview',background=[('selected','blue')])


#Style.configure('Information.Help.TButton')#,background='light blue',height=12)

def get_help():

    #will probabily need to specify the path like this tho
    #need the double (\\) for the path or errors will be thrown for some reason
    
    #example path if the one in the code does not work
    #path="C:\\Users\\jpste\\OneDrive\\Documents\\Python Scripts\\Spyder_ACUA\\ACUA_GUI\\HelpMenu_messages.pdf"
    path="HelpMenu_messages.pdf"
    os.startfile(path)
    
    
# ##################################################Tab formation############################################################
tabs = ttk.Notebook(ACUAgui,height=30, width=100)
#Create the tabs themselves
riskpipe = ttk.Frame(tabs,style='BLU.TFrame')
search = ttk.Frame(tabs,style='BLU.TFrame')
ModelBuild=ttk.Frame(tabs,style='BLU.TFrame')
#Help=ttk.Frame(tabs,style='BLU.TFrame')

#adds tab 
tabs.add(riskpipe, text= "Pipelines at Most Risk")
tabs.add(search, text= "Pipeline Database")
tabs.add(ModelBuild,text='Model Information')


tabs.pack(expand=1, fill="both") 

help_button=ttk.Button(tabs,text="Help Documentation",command=get_help)
help_button.pack(side='right',anchor='ne')

AP_button=ttk.Button(tabs,text="Asset Predictor Tool",command=pipeline_test_gui)
AP_button.pack(side='right',anchor='ne')


########################################### Tab1 == Piplelines at Most Risk ####################################################
##################################### Pipelines at Most Risk / Top Ten wdiget ##################################################
#Creates Label frame to place risktree onto the screen
RiskTreeF= ttk.LabelFrame(riskpipe, text="Most at Risk Assets")
RiskTreeF.place(relheight=0.35, relwidth=1, y=15)

MARToolbar=ttk.Frame(RiskTreeF)
MARToolbar.pack(side='top')#,anchor='center')

MARLabel=ttk.Label(MARToolbar,text='Enter how many assets you would like to see')
MARLabel.pack(side='left', fill='y',padx=5)

Pipe_num=StringVar()
input1entry= ttk.Entry(MARToolbar, textvariable=Pipe_num, width=30)
input1entry.pack(side='left')
Pipe_num.set('10')

def top_filter(n,dataframe):
    #clear treeview if anything is there
    trvwRisk.delete(*trvwRisk.get_children())
    #need to change the query when the right table is in the database
    MAR_df=dataframe.loc[:,['Pipe_ID', 'Town', 'Location_Info', 
                                   'Segment_Start','Segment_End','Random Forest OPMN', 
                                   'Years_Since_Last_Inspection','Remaining_Life']]
    MAR_df=MAR_df.sort_values('Random Forest OPMN',ascending=False)
    MAR_df_view=MAR_df.head(int(n))
    column_list=list(MAR_df.columns)
    column_names=[]
    for i in column_list:
        new_string=i.replace("_"," ")
        column_names.append(new_string)
    #assigns the treeview columns
    trvwRisk.configure(columns=column_names)
    #assigns the columns to the treeview
    for column in trvwRisk["columns"]:
        trvwRisk.heading(column, text=column)# let the column heading = column name
    #assigns the data in the rows of the treeview
    dataframe_rows = MAR_df_view.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in dataframe_rows:
        trvwRisk.insert("", "end", values=row)
    
#formation of treeview
trvwRisk= ttk.Treeview(RiskTreeF, show= 'headings')
trvwRisk.pack(padx=5, pady=5, fill='both', expand=True)
    
#Creates scoll bars in (x) directions
scrollbarx = ttk.Scrollbar(trvwRisk, orient="horizontal", command=trvwRisk.xview)
scrollbarx.pack(side="bottom", fill="x")
#Creates scoll bars in (y) directions
scrollbary=ttk.Scrollbar(trvwRisk, orient="vertical", command=trvwRisk.yview) #the command updates the yaxis of the parent widget
scrollbary.pack(side="right", fill="y")
#sets scroll bar (x/y)
trvwRisk.configure(xscrollcommand=scrollbarx.set, yscrollcommand=scrollbary.set)

#intiallizes the treeview when the app opens
top_filter(10,Pipe_DB_All)

# #Make a bind function to run RKFun
input1entry.bind("<Return>", lambda e: top_filter(Pipe_num.get(),Pipe_DB_All))


# ######################################## Pipelines at Most Risk / Report button ################################################
RiskF2=ttk.Frame(riskpipe) 
RiskF2.pack(padx=100,pady=250,side='bottom')


#Report Button
rptbut=ttk.Button(RiskF2, text="Generate Report", command= lambda: make_report(Model_ID,Pipe_DB_All))
rptbut.pack(padx=15, pady=4, fill='both')


################################################## Tab2 == search #############################################################
################################## Search / container 2 treeview search widget ##########################################



###bottom half of the tab
bottom_half=ttk.Frame(search,style='BLU.TFrame',height=500 )
bottom_half.pack(side='bottom',fill='both')
#Top Half of the tab
SearchTreeF= ttk.LabelFrame(search, text="Pipeline Database")
SearchTreeF.pack(side='top',fill='both', expand=True)

########################Creates toolbar frame#####################
toolbar=ttk.Frame(SearchTreeF)
toolbar.pack(side='top',fill='x')


# changing the row width based on what is contained in it
##https://stackoverflow.com/questions/39609865/how-to-set-width-of-treeview-in-tkinter-of-python
#Second dataframe for the search function

#global trvwSearch
trvwSearch= ttk.Treeview(SearchTreeF,show='headings',style='BW.Treeview')
trvwSearch.pack(padx=5, pady=5, fill='both', expand=True,side='top')


#Creates scoll bars in (x) directions    
scrollbarx2 = ttk.Scrollbar(SearchTreeF, orient="horizontal", command=trvwSearch.xview)
scrollbarx2.pack(side="bottom", fill="x")     
#Creates scoll bars in (y) directions  
scrollbary2=ttk.Scrollbar(trvwSearch, orient="vertical", command=trvwSearch.yview) 
scrollbary2.pack(side="right", fill="y") 
# assign the scrollbars to the Treeview Widget
trvwSearch.configure(xscrollcommand=scrollbarx2.set, yscrollcommand=scrollbary2.set) 



##################################creates the toolbar for the widgets above the treeveiw##################################

All_Database_columns=list(Pipe_DB_All.columns)
Relv_Database_columns=[e for e in All_Database_columns if e not in 
                                                   ('Town_ID',"Upstream_Pumpstation_ID",'Location', 'Population',
                                                    "Area_sqmi","Pop_Density","Source", "Recorded_date","Reported_pop/sqmi",
                                                    "Date","Flow_Orientation","Flow_Numerical_Category","Material",
                                                    "Mat_Numerical_Category","Pump_Sat_ID","G_F")]
searchOption=[]
for i in Relv_Database_columns:
    new_i=i.replace("_"," ")
    searchOption.append(new_i)
    
    
#drop down menu from all the columns in file
SelectS= StringVar()
dropSearch=ttk.OptionMenu(toolbar, SelectS, "Pipe ID",*searchOption)
dropSearch.config(width=35)
#SelectS.set("Town")
dropSearch.pack(side='left',anchor='nw')
        
#makes the search entry field
Search_variable=StringVar()
SearchEntry=ttk.Entry(toolbar,textvariable=Search_variable,text='put search field here',width=25)

SearchEntry.pack(side='left',fill='both', expand=True)


def search_df():
    global trvwSearch
    search_word=SearchEntry.get()
    Desired_Column=SelectS.get()
    Desired_Column=Desired_Column.replace(" ","_")
    search_query=[]
    raw_dataframe2=Pipe_DB_All
    Relv_dataframe=raw_dataframe2.loc[:,Relv_Database_columns]
    
    if search_word=="":
        TV_df=Relv_dataframe
    elif "." in search_word:
        search_query=float(search_word)
        TV_df=Relv_dataframe[Relv_dataframe[Desired_Column].isin([search_query])]
    else:
        try:
            search_query=int(search_word)
            TV_df=Relv_dataframe[Relv_dataframe[Desired_Column].isin([search_query])]
        except Exception:
            search_query=str(search_word)
            TV_df=Relv_dataframe[Relv_dataframe[Desired_Column].isin([search_query])]
    TV_df_rows = TV_df.to_numpy().tolist()
    
    if bool(TV_df_rows)==False:
       Errorgui=tk.Toplevel()
       Style=ttk.Style()
       Style.configure('Error.TLabel',foreground='red')
       w = 450 # width for the Tk root
       h = 50 # height for the Tk root
       ws = Errorgui.winfo_screenwidth() # width of the screen
       hs = Errorgui.winfo_screenheight() # height of the screen
       # calculate x and y coordinates for the Tk root window
       x = (ws/2) - (w/2)
       y = (hs/2) - (h/2)
       #set the dimensions of the screen  and where it is placed
       Errorgui.geometry('%dx%d+%d+%d' % (w, h, x, y-25))     
       label=ttk.Label(Errorgui,text="Erorr: please put in a valid entry",style ='Error.TLabel')
       label.pack(pady=10,padx=10)      
    else:
        trvwSearch.delete(*trvwSearch.get_children())
        column_list2=list(TV_df.columns)  
        column_names2=[]  
        for i in column_list2:
            new_string=i.replace("_"," ")
            column_names2.append(new_string)
        trvwSearch.configure(columns=column_names2)
        for column in trvwSearch["columns"]:
            trvwSearch.heading(column, text=column)# let the column heading = column name
        for i,row in enumerate(TV_df_rows):
            if i%2==0:
                trvwSearch.insert("", "end", values=row,tags=('oddrow',))
            else:
                trvwSearch.insert("", "end", values=row,tags=('evenrow',))    
        trvwSearch.tag_configure('evenrow',background='white')
        trvwSearch.tag_configure('oddrow',background='light grey')
        
search_df()



#makes the searh entry button
Search_Button=ttk.Button(toolbar,text='Search',width=25,command=search_df)
Search_Button.pack(side='left',fill='both')


def reset_df():
    global trvwSearch
    trvwSearch.delete(*trvwSearch.get_children())
    raw_dataframe2=Pipe_DB_All
    Relv_dataframe=raw_dataframe2.loc[:,Relv_Database_columns]
    TV_df_rows = Relv_dataframe.to_numpy().tolist()
    for i,row in enumerate(TV_df_rows):
        if i%2==0:
            trvwSearch.insert("", "end", values=row,tags=('evenrow',))
        else:
            trvwSearch.insert("", "end", values=row,tags=('oddrow',))
Reset_Button=ttk.Button(toolbar,text='Reset',width=25,command=reset_df)
Reset_Button.pack(side='left',fill='both')



SearchEntry.bind("<Return>", lambda e: search_df())

# ########################################## Search / Corrections Enter Changes  ############################################################
pie_width=400
LBH_w=w-pie_width
#frame for pie chart
Right_BH_frame=ttk.Frame(bottom_half, height= 275, width= pie_width, style='BLU.TFrame')
Right_BH_frame.pack(side='right',fill='both')

#frame for correction toolbar and all textboxes
Left_BH_frame=ttk.Frame(bottom_half,width= LBH_w, style='BLU.TFrame')
Left_BH_frame.pack(side='left', fill='both',expand=True)


Right_BH_frame.pack_propagate(False)

CorrectionToolbar=ttk.Frame(Left_BH_frame)
CorrectionToolbar.pack(pady=5,side='top')


OtherTV= ttk.Treeview(Left_BH_frame,show='headings',style='BW.Treeview')
OtherTV.pack(padx=5, pady=2, fill='both', expand=True,side='right')

# label frame for all entry fields and labels
SearchF3=ttk.LabelFrame(Left_BH_frame, text="Asset Information")
SearchF3.pack(pady=5,side='top',fill='x',expand='true')

pady_val=3
#town label and entry box
Town_label = ttk.Label(SearchF3, text="Town")
Town_label.grid(row=0, column=0, padx=10, pady=pady_val)
Town_entry = ttk.Entry(SearchF3)
Town_entry.grid(row=0, column=1, padx=10, pady=pady_val)
Town_entry.config(state='disabled')

#Segment Start label and entry box
SS_label = ttk.Label(SearchF3, text="Segment Start")
SS_label.grid(row=0, column=2, padx=10, pady=pady_val)
SS_entry = ttk.Entry(SearchF3)
SS_entry.grid(row=0, column=3, padx=10, pady=pady_val)
SS_entry.config(state='disabled')

#Segment End label and entry box
SE_label = ttk.Label(SearchF3, text="Segment End")
SE_label.grid(row=0, column=4, padx=10, pady=pady_val)
SE_entry = ttk.Entry(SearchF3)
SE_entry.grid(row=0, column=5, padx=10, pady=pady_val)
SE_entry.config(state='disabled')

#Pipe Diameter label and entry box
diam_label = ttk.Label(SearchF3, text="Diameter (in)")
diam_label.grid(row=1, column=0, padx=10, pady=pady_val)
diam_entry = ttk.Entry(SearchF3)
diam_entry.grid(row=1, column=1, padx=10, pady=pady_val)
diam_entry.config(state='disabled')

#Pipe Length label and entry box
len_label = ttk.Label(SearchF3, text="Length (ft)")
len_label.grid(row=1, column=2, padx=10, pady=pady_val)
len_entry = ttk.Entry(SearchF3)
len_entry.grid(row=1, column=3, padx=10, pady=pady_val)
len_entry.config(state='disabled')

#Pipe type label and entry box
PT_label = ttk.Label(SearchF3, text="Pipe Type (Material)")
PT_label.grid(row=1, column=4, padx=10, pady=pady_val)
PT_entry = ttk.Entry(SearchF3)
PT_entry.grid(row=1, column=5, padx=10, pady=pady_val)
PT_entry.config(state='disabled')

#G/F label and entry box
GF_label = ttk.Label(SearchF3, text="Gravity or Force Main")
GF_label.grid(row=2, column=0, padx=10, pady=pady_val)
GF_entry = ttk.Entry(SearchF3)
GF_entry.grid(row=2, column=1, padx=10, pady=pady_val)
GF_entry.config(state='disabled')

#Original Installation label and entry box
OI_label = ttk.Label(SearchF3, text="Original Installation")
OI_label.grid(row=2, column=2, padx=10, pady=pady_val)
OI_entry = ttk.Entry(SearchF3)
OI_entry.grid(row=2, column=3, padx=10, pady=pady_val)
OI_entry.config(state='disabled')

#Remaining Life label and entry box
RML_label = ttk.Label(SearchF3, text="Remaining Life (years)")
RML_label.grid(row=2, column=4, padx=10, pady=pady_val)
RML_entry = ttk.Entry(SearchF3)
RML_entry.grid(row=2, column=5, padx=10, pady=pady_val)
RML_entry.config(state='disabled')

#Last inspection label and entry box
LINS_label = ttk.Label(SearchF3, text="Years Since Last Inspection")
LINS_label.grid(row=3, column=0, padx=10, pady=pady_val)
LINS_entry = ttk.Entry(SearchF3)
LINS_entry.grid(row=3, column=1, padx=10, pady=pady_val)
LINS_entry.config(state='disabled')

SearchF4=ttk.LabelFrame(Left_BH_frame, text="Asset Scores")
SearchF4.pack(side='top',fill='x',expand='true')

TRP_label = ttk.Label(SearchF4, text="True Risk Probability")
TRP_label.grid(row=0, column=0, padx=10, pady=pady_val)
TRP_entry = ttk.Entry(SearchF4)
TRP_entry.grid(row=0, column=1, padx=10, pady=pady_val)
TRP_entry.config(state='disabled')

TFI_label = ttk.Label(SearchF4, text="True Failure Impact")
TFI_label.grid(row=0, column=2, padx=10, pady=pady_val)
TFI_entry = ttk.Entry(SearchF4)
TFI_entry.grid(row=0, column=3, padx=10, pady=pady_val)
TFI_entry.config(state='disabled')

TRF_label = ttk.Label(SearchF4, text="True Risk Factor")
TRF_label.grid(row=0, column=4, padx=10, pady=pady_val)
TRF_entry = ttk.Entry(SearchF4)
TRF_entry.grid(row=0, column=5, padx=10, pady=pady_val)
TRF_entry.config(state='disabled')

RFRP_label = ttk.Label(SearchF4, text="Model Risk Probability")
RFRP_label.grid(row=1, column=0, padx=10, pady=pady_val)
RFRP_entry = ttk.Entry(SearchF4)
RFRP_entry.grid(row=1, column=1, padx=10, pady=pady_val)
RFRP_entry.config(state='disabled')

RFFI_label = ttk.Label(SearchF4, text="Model Failure Impact")
RFFI_label.grid(row=1, column=2, padx=10, pady=pady_val)
RFFI_entry = ttk.Entry(SearchF4)
RFFI_entry.grid(row=1, column=3, padx=10, pady=pady_val)
RFFI_entry.config(state='disabled')

RFRF_label = ttk.Label(SearchF4, text="Model Risk Factor")
RFRF_label.grid(row=1, column=4, padx=10, pady=pady_val)
RFRF_entry = ttk.Entry(SearchF4)
RFRF_entry.grid(row=1, column=5, padx=10, pady=pady_val)
RFRF_entry.config(state='disabled')

#creates the function of what happens when a record is selected
def select_record(e):
    try:
        #Grab record number
        selected = trvwSearch.focus()
        #grab record values
        values=trvwSearch.item(selected, 'values')
        
        desired_model=Model_ID.get()
        desired_df=model_id_tab.loc[model_id_tab["Model_name"]==desired_model,['Model_ID']]
        desired_index=desired_df.iloc[0,0]
        
        RP_score=values[26]
        FI_score=values[27]
        
        for child in Right_BH_frame.winfo_children():
            child.destroy()
        
        WSA_model=WSA(int(RP_score),int(FI_score),desired_index)  
        figure=WSA_model.figure  
        OPMNCanvas = FigureCanvasTkAgg(figure,master=Right_BH_frame)
        OPMNCanvas.draw()
        OPMNCanvas.get_tk_widget().pack()
        
        
        OtherTV.delete(*OtherTV.get_children())
        dataframe=WSA_model.other_labels
        dataframe_rows = dataframe.to_numpy().tolist() # turns the dataframe into a list of lists
        TV_width=OtherTV.winfo_width()-2
        OtherTV.configure(columns=('Factors in Other Category',))
        for column in OtherTV["columns"]:
            OtherTV.heading(column, text=column)# let the column heading = column name
        OtherTV.column(0,width=TV_width)
        for i,row in enumerate(dataframe_rows):
            if i%2==0:
                OtherTV.insert("", "end", values=row,tags=('evenrow',))
            else:
                OtherTV.insert("", "end", values=row,tags=('oddrow',))


        Dropselection=dropCorrection._variable.get()
    
        txtboxes=[Town_entry,SS_entry,SE_entry,
                  diam_entry,len_entry,PT_entry,
                  GF_entry,OI_entry,RML_entry,LINS_entry,
                  TRP_entry,TFI_entry,TRF_entry,RFRP_entry,
                  RFFI_entry,RFRF_entry]
    
        for i in txtboxes:
            i.config(state="normal")
        
    #clear entry boxes
        for i in txtboxes:
            i.delete(0,'end')


    #output values - Asset Information
        Town_entry.insert(0,values[1])
        SS_entry.insert(0,values[3])
        SE_entry.insert(0,values[4])
        diam_entry.insert(0,values[9])
        len_entry.insert(0,values[7])
        PT_entry.insert(0,values[8])
        GF_entry.insert(0,values[5])
        OI_entry.insert(0,values[10])
        RML_entry.insert(0,values[25])
        LINS_entry.insert(0,values[24])
    #output values - Asset Scores
        TRP_entry.insert(0,values[17])
        TFI_entry.insert(0,values[16])
        TRF_entry.insert(0,values[18])
        RFRP_entry.insert(0,values[26])
        RFFI_entry.insert(0,values[27])
        RFRF_entry.insert(0,values[28])
    
        if Dropselection=='View data':
            for i in txtboxes:
                i.config(state="disable")
            
        #elif Dropselection=='Add New Asset':
        #    for i in txtboxes:
        #        i.delete(0,'end')
        elif Dropselection=='Material Change':
            txtboxes.remove(PT_entry)
            for i in txtboxes:
                i.config(state="disable")
        elif Dropselection=='Diameter Change':
            txtboxes.remove(diam_entry)
            for i in txtboxes:
                i.config(state="disable")   
        elif Dropselection=='Length Change':
            txtboxes.remove(len_entry)
            for i in txtboxes:
                i.config(state="disable") 
        elif Dropselection=='True Score Change':
            txtboxes.remove(TRP_entry)
            txtboxes.remove(TFI_entry)
            txtboxes.remove(TRF_entry)
            for i in txtboxes:
                i.config(state="disable")             
        elif Dropselection=='Alter Pipeline Data':
            txtboxes.remove(diam_entry)
            
    except IndexError:
        A=0

trvwSearch.bind("<ButtonRelease-1>", select_record)

###################################        Toolbar for corrections    #######################################################
#delete statement

def delete_data():
    Checkgui=tk.Toplevel()
    
    def yes_command_dele():
        user='root' #user name for the user that is connecting to the database
        password='password' #password for the user that is connecting to the database
        host='localhost' #host name or ip address, for local machines it is localhost otherwise it is the server ip 
        database='test_acua_data' #database which you are connecting to
        
        my_conn=mysql.connector.connect(host=host,
                                         user=user,
                                         password=password,
                                         database=database)    
        mycursor=my_conn.cursor()
        
        selected = trvwSearch.focus()
        values=list(trvwSearch.item(selected, 'values'))
        drop_index=values[0]
    
        #delete from treeview
        selected_item = trvwSearch.selection() ## get selected item
        trvwSearch.delete(selected_item)
        
        #delete from dataframe
        drop_df=[]
        for i in range(len(Pipe_DB_All)):
            if Pipe_DB_All.iloc[i,0]==int(drop_index):
                drop_df.append(i)
                
        Pipe_DB_All.drop(drop_df,axis=0,inplace=True)
        Pipe_DB_All.reset_index(drop=True,inplace=True)
        
        #delete from database
        drop_query="""DELETE FROM `test_acua_data`.`pipeline_database` WHERE (`Pipe_ID` = '%s');"""%str(drop_index)
        mycursor.execute(drop_query)
        my_conn.commit()
        my_conn.close()
        Checkgui.destroy()

    def no_command_dele():
        Checkgui.destroy()
        
    yesButton=ttk.Button(Checkgui,text='Yes, I want to delete this asset', command= yes_command_dele)
    yesButton.pack(pady=10,padx=20)
    noButton=ttk.Button(Checkgui,text='I do not want to delete this asset',command=no_command_dele)
    noButton.pack(pady=10,padx=20)
        
        
#if dropdown= 'Change all data' --> make a popup where they can change any data
#else do the current thing
def change_data():
    user='root' #user name for the user that is connecting to the database
    password='password' #password for the user that is connecting to the database
    host='localhost' #host name or ip address, for local machines it is localhost otherwise it is the server ip 
    database='test_acua_data' #database which you are connecting to
    
    my_conn=mysql.connector.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database)    
    mycursor=my_conn.cursor()
    
    selected = trvwSearch.focus()
    values=list(trvwSearch.item(selected, 'values'))
    
    pipe_id=values[0]
    values[1]=Town_entry.get()
    values[3]=SS_entry.get()
    values[4]=SE_entry.get()
    values[9]=diam_entry.get()
    values[7]=len_entry.get()
    values[8]=PT_entry.get()
    values[5]=GF_entry.get()
    values[10]=OI_entry.get()
    values[25]=RML_entry.get()
    values[24]=LINS_entry.get()
    values[17]=TRP_entry.get()
    values[16]=TFI_entry.get()
    values[18]=TRF_entry.get()
    values[26]=RFRP_entry.get()
    values[27]=RFFI_entry.get()
    values[28]=RFRF_entry.get()
    
    trvwSearch.item(selected, text="", values=values)
    update_query="""UPDATE `test_acua_data`.`pipeline_database` SET 
                                                        `Town` = '%s', 
                                                        `Segment_Start` = '%s', 
                                                        `Segment_End` = '%s', 
                                                        `Force_or_Gravity_Main` = '%s', 
                                                        `Segment_Length_ft` = '%s', 
                                                        `Pipe_Type` = '%s', 
                                                        `G_F` = '%s', 
                                                        `Pipe_Size_in` = '%s', 
                                                        `Original_Installation` = '%s', 
                                                        `Last_Inspection` = '%s', 
                                                        `True_Failure_Impact` = '%s', 
                                                        `True_Risk_Probability` = '%s', 
                                                        `Risk_Factor` = '%s' 
                                                        WHERE (`Pipe_ID` = '%s');""" %(Town_entry.get(),
                                                        SS_entry.get(),
                                                        SE_entry.get(),
                                                        GF_entry.get(),
                                                        len_entry.get(),
                                                        PT_entry.get(),
                                                        GF_entry.get(),
                                                        diam_entry.get(),
                                                        OI_entry.get(),
                                                        LINS_entry.get(),
                                                        TFI_entry.get(),
                                                        TRP_entry.get(),
                                                        TRF_entry.get(),
                                                        pipe_id)
    mycursor.execute(update_query)
    my_conn.commit()
    my_conn.close()
# =====================================Dataframe column Positions========================================
#     1=Town_entry.get()
#     4=SS_entry.get()
#     5=SE_entry.get()
#     12=diam_entry.get()
#     9=len_entry.get()
#     10=PT_entry.get()
#     6=GF_entry.get()
#     11=GF_entry.get()
#     13=OI_entry.get()
#     42=RML_entry.get()
#     41=LINS_entry.get()
#     20=TRP_entry.get()
#     19=TFI_entry.get()
#     21=TRF_entry.get()
#     43=RFRP_entry.get()
#     44=RFFI_entry.get()
#     45=RFRF_entry.get()
# =============================================================================
# =============================================================================
#     index_list=[1,4,5,12,9,10,6,11,13,42,41,20,19,21,43,44,45]
#     entry_list=[Town_entry,SS_entry,SE_entry,
#            diam_entry,len_entry,PT_entry,
#            GF_entry,GF_entry,OI_entry,RML_entry,LINS_entry,
#            TRP_entry,TFI_entry,TRF_entry,RFRP_entry,
#            RFFI_entry,RFRF_entry]
#     for i,j in zip(index_list,entry_list):
#         Pipe_DB_All.iloc[int(pipe_id),i]=j.get()
# =============================================================================

def add_pipe():    
    APgui=tk.Toplevel()
    APgui.title("Add a pipeline")
    
    
    user='root' #user name for the user that is connecting to the database
    password='password' #password for the user that is connecting to the database
    host='localhost' #host name or ip address, for local machines it is localhost otherwise it is the server ip 
    database='test_acua_data' #database which you are connecting to

    engine = create_engine("mysql+pymysql://%s:%s@%s/%s" %(user,
                                                           password,
                                                           host,
                                                           database))


    width_of_drop=15
    #All location information pertaining to pipeline
    location_label=ttk.Label(APgui,text='Pipline Location Information',style='Head.TLabel')
    location_label.grid(row=0,column=0,columnspan=6,sticky='w')
    #town infromation
    Town_label=ttk.Label(APgui,text='Town of Pipeline')
    Town_label.grid(row=1,column=0, padx=5, pady=5)
    town_query='SELECT Location FROM pop_data'
    townvalues= pd.read_sql(town_query,con=engine)
    town_opt=list(townvalues['Location']) 
    town_var=StringVar()
    town_drop=ttk.OptionMenu(APgui,town_var,town_opt[0],*town_opt)
    town_drop.config(width=width_of_drop)
    town_drop.grid(row=1, column=1, padx=5, pady=5)
    #segment start info
    SS_label=ttk.Label(APgui,text='Segment Start of Pipeline')
    SS_label.grid(row=1,column=2, padx=5, pady=5)
    SS_entry=ttk.Entry(APgui)
    SS_entry.grid(row=1,column=3, padx=5, pady=5)      
    #segment start info
    SE_label=ttk.Label(APgui,text='Segment End of Pipeline')
    SE_label.grid(row=1,column=4, padx=5, pady=5)
    SE_entry=ttk.Entry(APgui)
    SE_entry.grid(row=1,column=5, padx=5, pady=5)          
    #location information
    loc_label=ttk.Label(APgui,text='Location information of Pipeline')
    loc_label.grid(row=2,column=0, padx=5, pady=5)
    loc_entry=ttk.Entry(APgui)
    loc_entry.config(width=35)
    loc_entry.grid(row=2,column=1, padx=5, pady=5,columnspan=5,sticky='ew')  
    
    #All pipeline Properties 
    prop_label=ttk.Label(APgui,text='Pipline Properties and Dimension Information',style='Head.TLabel')
    prop_label.grid(row=3,column=0,columnspan=6,sticky='w')
    #force/grav
    FG_label=ttk.Label(APgui,text='Flow by Force or Gravity?')
    FG_label.grid(row=4,column=0, padx=5, pady=5)
    FG_query='SELECT Flow_Orientation FROM flow_category'
    FG_values= pd.read_sql(FG_query,con=engine)
    FG_opt=list(FG_values['Flow_Orientation']) 
    FG_var=StringVar()
    FG_drop=ttk.OptionMenu(APgui,FG_var,FG_opt[0],*FG_opt)
    FG_drop.config(width=width_of_drop)
    FG_drop.grid(row=4, column=1, padx=5, pady=5)
    #pumpstation 
    pumpst_label=ttk.Label(APgui,text='Select Upstream Pumpstation:')
    pumpst_label.grid(row=4,column=2, padx=5, pady=5)
    pumpst_query='SELECT Pump_Station FROM pump_stat_avg'
    pumpst_values= pd.read_sql(pumpst_query,con=engine)
    pumpst_opt=list(pumpst_values['Pump_Station']) 
    pumpst_var=StringVar()
    pumpst_drop=ttk.OptionMenu(APgui,pumpst_var,pumpst_opt[0],*pumpst_opt)
    pumpst_drop.config(width=width_of_drop)
    pumpst_drop.grid(row=4, column=3, padx=5, pady=5)    
    #segment length
    SL_label=ttk.Label(APgui,text='Segment Length of Pipeline:')
    SL_label.grid(row=4,column=4, padx=5, pady=5)
    SL_entry=ttk.Entry(APgui)
    SL_entry.grid(row=4,column=5, padx=5, pady=5)     
    #pipe material
    mat_label=ttk.Label(APgui,text='Select Pipe Material:')
    mat_label.grid(row=5,column=0, padx=5, pady=5)
    mat_query='SELECT Material FROM material_category'
    mat_values= pd.read_sql(mat_query,con=engine)
    mat_opt=list(mat_values['Material']) 
    mat_var=StringVar()
    mat_drop=ttk.OptionMenu(APgui,mat_var,mat_opt[0],*mat_opt)
    mat_drop.config(width=width_of_drop)
    mat_drop.grid(row=5, column=1, padx=5, pady=5)     
    #pipe diameter
    PD_label=ttk.Label(APgui,text='Pipeline Diameter:')
    PD_label.grid(row=5,column=2, padx=5, pady=5)
    PD_entry=ttk.Entry(APgui)
    PD_entry.grid(row=5,column=3, padx=5, pady=5)     
    #up/downstream
    UD_label=ttk.Label(APgui,text='Pipeline Placement Value')
    UD_label.grid(row=5,column=4, padx=5, pady=5)
    UD_entry=ttk.Entry(APgui)
    UD_entry.grid(row=5,column=5, padx=5, pady=5)     
    
    #installation properties
    install_label=ttk.Label(APgui,text='Installation and Rehabilitation Information',style='Head.TLabel')
    install_label.grid(row=6,column=0,columnspan=6,sticky='w')
    #original installation
    OI_label=ttk.Label(APgui,text='Original Installation')
    OI_label.grid(row=7,column=0, padx=5, pady=5)
    OI_entry=ttk.Entry(APgui)
    OI_entry.grid(row=7,column=1, padx=5, pady=5)     
    #original lifespan
    OL_label=ttk.Label(APgui,text='Original Lifespan')
    OL_label.grid(row=7,column=2, padx=5, pady=5)
    OL_entry=ttk.Entry(APgui)
    OL_entry.grid(row=7,column=3, padx=5, pady=5)     
    #year of rehabilation or replacement
    RehabY_label=ttk.Label(APgui,text='Year of Rehabilitation or Replacement')
    RehabY_label.grid(row=7,column=4, padx=5, pady=5)
    RehabY_entry=ttk.Entry(APgui)
    RehabY_entry.grid(row=7,column=5, padx=5, pady=5)    
    #added design life from rehabilitation
    RehabAdd_label=ttk.Label(APgui,text='Added Design Life from Rehabilitation [Years]')
    RehabAdd_label.grid(row=8,column=0, padx=5, pady=5)
    RehabAdd_entry=ttk.Entry(APgui)
    RehabAdd_entry.grid(row=8,column=1, padx=5, pady=5)     
    #rehabilitation type
    RehabTy_label=ttk.Label(APgui,text='Rehabilitation Type')
    RehabTy_label.grid(row=8,column=2, padx=5, pady=5)
    RehabTy_entry=ttk.Entry(APgui)
    RehabTy_entry.grid(row=8,column=3, padx=5, pady=5)     
    #last inspection
    LI_label=ttk.Label(APgui,text='Year pipe was last inspected')
    LI_label.grid(row=8,column=4, padx=5, pady=5)
    LI_entry=ttk.Entry(APgui)
    LI_entry.grid(row=8,column=5, padx=5, pady=5)   
    
    
    
    #Score values
    score_label=ttk.Label(APgui,text='Pipeline Scores',style='Head.TLabel')
    score_label.grid(row=9,column=0,columnspan=6,sticky='w')
    #Failure impact
    FI_label=ttk.Label(APgui,text='Failure Impact Score')
    FI_label.grid(row=10,column=0, padx=5, pady=5)
    FI_entry=ttk.Entry(APgui)
    FI_entry.grid(row=10,column=1, padx=5, pady=5) 
    #risk probability
    RP_label=ttk.Label(APgui,text='Risk Probability Score')
    RP_label.grid(row=10,column=2, padx=5, pady=5)
    RP_entry=ttk.Entry(APgui)
    RP_entry.grid(row=10,column=3, padx=5, pady=5) 
    #Risk factor
    RF_label=ttk.Label(APgui,text='Risk Factor Score')
    RF_label.grid(row=10,column=4, padx=5, pady=5)
    RF_entry=ttk.Entry(APgui)
    RF_entry.grid(row=10,column=5, padx=5, pady=5) 
    #subsequent data
    sd_label=ttk.Label(APgui,text='Other Needed Information',style='Head.TLabel')
    sd_label.grid(row=11,column=0,columnspan=6,sticky='w')
    #not ACUA
    NACUA_label=ttk.Label(APgui,text='Is the pipeline not owned by the ACUA?')
    NACUA_label.grid(row=12,column=0, padx=5, pady=5)
    NACUA_opt=list(['Yes','No']) 
    NACUA_var=StringVar()
    NACUA_drop=ttk.OptionMenu(APgui,NACUA_var,NACUA_opt[0],*NACUA_opt)
    NACUA_drop.config(width=width_of_drop)
    NACUA_drop.grid(row=12, column=1, padx=5, pady=5) 
    #abandoned 
    abd_label=ttk.Label(APgui,text='Is the pipeline abandoned?')
    abd_label.grid(row=12,column=2, padx=5, pady=5)
    abd_opt=list(['Yes','No']) 
    abd_var=StringVar()
    abd_drop=ttk.OptionMenu(APgui,abd_var,abd_opt[0],*abd_opt)
    abd_drop.config(width=width_of_drop)
    abd_drop.grid(row=12, column=3, padx=5, pady=5)    
    #any notes
    note_label=ttk.Label(APgui,text='Pipeline Notes:')
    note_label.grid(row=13,column=0, padx=5, pady=5)
    note_entry=ttk.Entry(APgui)
    note_entry.config(width=35)
    note_entry.grid(row=14,column=0, padx=5, pady=5,columnspan=6,sticky='ew')      
    

    def add_to_db():
        user='root' #user name for the user that is connecting to the database
        password='password' #password for the user that is connecting to the database
        host='localhost' #host name or ip address, for local machines it is localhost otherwise it is the server ip 
        database='test_acua_data' #database which you are connecting to
    
        my_conn=mysql.connector.connect(host=host,
                                         user=user,
                                         password=password,
                                         database=database)
        mycursor=my_conn.cursor()
        
        engine = create_engine("mysql+pymysql://%s:%s@%s/%s" %(user,
                                                           password,
                                                           host,
                                                           database))
               
        query="""SELECT * FROM test_acua_data.pipeline_database"""
        pipe_db=pd.read_sql(query,con=engine)
        new_id=str(pipe_db.iloc[-1,0]+1)
        db_columns=pipe_db.columns
        
        column_list_not_joined=[]
        for name in db_columns:
            column_list_not_joined.append(name)

        column_list=", ".join([str(item) for item in column_list_not_joined])

        town=town_var.get()
        town_query="""SELECT Town_ID FROM test_acua_data.pop_data WHERE Location='%s'"""%town
        town_db=pd.read_sql(town_query,con=engine)
        town_ID=str(town_db.iloc[0,0])
        
        
        PS=pumpst_var.get()
        PS_query="""SELECT Pump_Sat_ID FROM test_acua_data.pump_stat_avg WHERE Pump_Station='%s'"""%PS
        PS_db=pd.read_sql(PS_query,con=engine)
        PS_ID=str(PS_db.iloc[0,0])
        var_w_vals=[new_id,town_ID,PS_ID]
        var_w_YN=[NACUA_var,abd_var]
        var_order=[new_id,town_var,town_ID,loc_entry,SS_entry,SE_entry,FG_var,
                   PS_ID,UD_entry,SL_entry,mat_var,FG_var, PD_entry,
                   OI_entry,OL_entry,RehabY_entry,
                   RehabAdd_entry,RehabTy_entry,
                   LI_entry,FI_entry,RP_entry,
                   RF_entry,NACUA_var,abd_var,note_entry]
        vals_to_insert=[]
        for i in var_order:
            if i in var_w_vals:
                vals_to_insert.append(i)
            elif i in var_w_YN:
                if i.get() =="Yes":
                    vals_to_insert.append('1')
                else:
                    vals_to_insert.append('0')
            else:
                vals_to_insert.append(i.get())     
        insert_list=str(vals_to_insert)[1:-1]
        add_query="""INSERT INTO test_acua_data.pipeline_database (%s) VALUES (%s);"""%(column_list,insert_list)
        mycursor.execute(add_query)
        my_conn.commit()
        my_conn.close()
        make_pipe_db()
        top_filter(10,Pipe_DB_All)
        search_df()

    Execute_Button=ttk.Button(APgui,text='Add Pipeline to Database',width=25,command=add_to_db)
    Execute_Button.grid(row=15, column=0, padx=2, pady=2,sticky='ew',columnspan=6)

    APgui.mainloop()    
    
CorrectionLabel=ttk.Label(CorrectionToolbar,text='Select the correction', style='BLU.TLabel')
CorrectionLabel.pack(side='left', fill='both')

#https://stackoverflow.com/questions/54641750/python-typeerror-lambda-takes-0-positional-arguments-but-1-was-given-due-t?rq=1
CorrectionOption= ['View data','Material Change','Diameter Change','Length Change','True Score Change','Alter Pipeline Data']
CorrectionSelect= StringVar()
dropCorrection=ttk.OptionMenu(CorrectionToolbar,CorrectionSelect, CorrectionOption[0],*CorrectionOption,command=lambda e: select_record(e))
dropCorrection.config(width=20)
dropCorrection.pack(side='left')
#bind the treeview to function select_record


#creates button labeled populate data
change_button=ttk.Button(CorrectionToolbar,text="Change Data",command=change_data)#,command=select_record)
change_button.pack(side='left')

deleteRecord_button=ttk.Button(CorrectionToolbar,text="Delete Asset",command=delete_data)
deleteRecord_button.pack(side='left')

addRecord_button=ttk.Button(CorrectionToolbar,text="Add Asset",command=add_pipe)#,command=select_record)
addRecord_button.pack(side='left')

###
select_asset_LB=ttk.Label(Right_BH_frame, text='select an asset to see pie chart')
select_asset_LB.pack(pady=5)


# #############################################Tab 3 == Model #############################################################


New_model_feat_RP=pd.DataFrame()
New_model_feat_FI=pd.DataFrame()
def run_new_model(treeview, frame,dropdown,model=['RP','FI']):
    global New_model_feat_RP,New_model_feat_FI
    factors=[]
    
    for row in treeview.get_children():
        values=treeview.item(row)['values'][0]
        factors.append(values)

    new_lst=(','.join( repr(e) for e in factors ))
    query=('SELECT Factors, Factor_label FROM factors where Factor_label in (%s) and Model_ID=%s;'%(new_lst,str(saved_model)) )       
    
    fact=pd.read_sql(query,con=engine)
    factors=fact["Factors"]
    factor_labels=list(fact["Factor_label"])
    if model=="RP":
        New_model_feat_RP=[]
        dependentvariable='True_Risk_Probability'
        New_model=make_model(pipe_DB,factors=factors,dependentvariable=dependentvariable,labels=factor_labels)
        New_model_feat_RP=New_model.feature_importances
        New_RP_acc_entry.config(state="normal")
        New_RP_acc_entry.delete(0,'end')
        New_RP_acc_entry.insert(0,"{:.2f}".format(New_model.accuracy_perc))
        New_RP_acc_entry.config(state="disabled")
    elif model == "FI":
        New_model_feat_FI=[]
        dependentvariable='True_Failure_Impact'
        New_model=make_model(pipe_DB,factors=factors,dependentvariable=dependentvariable,labels=factor_labels)
        New_model_feat_FI=New_model.feature_importances
        New_FI_acc_entry.config(state="normal")
        New_FI_acc_entry.delete(0,'end')
        New_FI_acc_entry.insert(0,"{:.2f}".format(New_model.accuracy_perc))
        New_FI_acc_entry.config(state="disabled")

    for child in frame.winfo_children():
        child.destroy()
    Dropselection=dropdown._variable.get()
    if Dropselection in 'Confusion Matrix': 
        NewCanvas = FigureCanvasTkAgg(New_model.conf_matrix,master=frame )
        NewCanvas.draw()
        NewCanvas.get_tk_widget().pack()        
    elif Dropselection in 'Feature Importance':
        NewCanvas = FigureCanvasTkAgg(New_model.plot,master=frame )
        NewCanvas.draw()
        NewCanvas.get_tk_widget().pack()
        
        
def change_graph(e):
    global BaseRP_Frame,BaseFI_Frame,NewRP_Frame,NewFI_Frame
    
    Dropselection=dropModel._variable.get()
    
    if Dropselection in 'Confusion Matrix':
        for i in [BaseRP_Frame,BaseFI_Frame,NewRP_Frame,NewFI_Frame]:
            for child in i.winfo_children():
                child.destroy()
        
        Base_RPCanvas = FigureCanvasTkAgg(RP_model.conf_matrix,master=BaseRP_Frame )
        Base_RPCanvas.draw()
        Base_RPCanvas.get_tk_widget().pack()
        
        Base_FICanvas = FigureCanvasTkAgg(FI_model.conf_matrix,master=BaseFI_Frame )
        Base_FICanvas.draw()
        Base_FICanvas.get_tk_widget().pack()
        run_new_model(trvwNewRP,NewRP_Frame,dropModel,model='RP')
        run_new_model(trvwNewFI,NewFI_Frame,dropModel,model='FI')
        
    elif Dropselection in 'Feature Importance':
        for i in [BaseRP_Frame,BaseFI_Frame,NewRP_Frame,NewFI_Frame]:
            for child in i.winfo_children():
                child.destroy()
            
        Base_RPCanvas = FigureCanvasTkAgg(RP_model.plot,master=BaseRP_Frame )
        Base_RPCanvas.draw()
        Base_RPCanvas.get_tk_widget().pack()
        
        Base_FICanvas = FigureCanvasTkAgg(FI_model.plot,master=BaseFI_Frame )
        Base_FICanvas.draw()
        Base_FICanvas.get_tk_widget().pack()
        
        run_new_model(trvwNewRP,NewRP_Frame,dropModel,model='RP')
        run_new_model(trvwNewFI,NewFI_Frame,dropModel,model='FI')      
    
def run_both_new_model():
    run_new_model(trvwNewRP,NewRP_Frame,dropModel,model='RP')
    run_new_model(trvwNewFI,NewFI_Frame,dropModel,model='FI')
    
def save_new_model():

    if New_model_feat_RP.empty == True or New_model_feat_FI.empty == True:
        Savegui=tk.Toplevel()
        Savegui.title("Error!")
        label=ttk.Label(Savegui,text='You must run new models for both Failure Impact and Risk Probability\n to save a new model!',
                        style='Error.TLabel')
        label.configure(anchor='center')
        label.pack(pady=5,padx=5)
    else:
        Model_ID_query=""" SELECT 
         *
       FROM
         model_id"""
        model_id_tab=pd.read_sql(Model_ID_query,con=engine)
        feat_imp_ind_RP=New_model_feat_RP.index.values.tolist()
        feat_imp_ind_FI=New_model_feat_FI.index.values.tolist()        
        save_loc=Save_ID.get()  
        db_sel=model_id_tab[model_id_tab['Model_name'].isin([save_loc])]
        sel_ID=list(db_sel.iloc[0])
        save_sel=str(sel_ID[0])
        save_query="""SELECT * From factors WHERE Model_ID=%s""" %save_sel
        fac_table=pd.read_sql(save_query,con=engine)
        print(fac_table)
        fact_list=['Years_Since_Last_Inspection','Average_Flow','Pipe_Size_in',
               'Pop_Density','Flow_Numerical_Category','Up_down_stream',
               'Mat_Numerical_Category','Remaining_Life','Segment_Length_ft','Original_Installation']
        info_list_RP=[]
        info_list_FI=[]
        print(len(fact_list))
        for i in fact_list:
            fact_info=[]
            new_selec=fac_table[fac_table['Factors'].isin([i])]
            new_selec.reset_index(drop=True,inplace=True)
            save_selec=str(new_selec.iloc[0,0])
            fact_info.append(i)
            fact_info.append(save_selec)
            if i in feat_imp_ind_RP:
                fact_info.append(New_model_feat_RP.loc[i,'percentage'])
            else:
                fact_info.append(0)            
            info_list_RP.append(fact_info)
        for i in fact_list:  
            fact_info=[]
            new_selec=fac_table[fac_table['Factors'].isin([i])]
            save_selec=str(new_selec.iloc[0,0])
            fact_info.append(i)
            fact_info.append(save_selec)
            if i in feat_imp_ind_FI:
                fact_info.append(New_model_feat_FI.loc[i,'percentage'])
            else:
                fact_info.append(0)            
            info_list_FI.append(fact_info)         
        OPMN_vals=[]    
        for i in fact_list:
            OPMN_info=[]
            if i in feat_imp_ind_RP:
                RP_score=New_model_feat_RP.loc[i,'percentage']
            else:
                RP_score=0
            if i in feat_imp_ind_FI:
                FI_score=New_model_feat_FI.loc[i,'percentage']
            else:
                FI_score=0   
            new_selec=fac_table[fac_table['Factors'].isin([i])]
            save_selec=str(new_selec.iloc[0,0])
            OPMN_info.append(save_selec)
            average_score=(RP_score+FI_score)/2
            OPMN_info.append(average_score)
            OPMN_vals.append(OPMN_info)
        
        New_model_name=str(SavingEntry.get())
        
        Checkgui=tk.Toplevel()

        def yes_command():
            model_update_query="UPDATE `test_acua_data`.`model_id` SET `Model_name` = '%s' WHERE (`Model_ID` = %s);" %(New_model_name,int(save_sel))
            mycursor.execute(model_update_query)
            my_conn.commit()
            for i,j,k in info_list_RP:
               queryRP="UPDATE `test_acua_data`.`factors` SET `RP_Importance` = %s WHERE (`factors_id` = %s); "%(k,j)
               mycursor.execute(queryRP)
               my_conn.commit()
            for i,j,k in info_list_FI:
               queryFI="UPDATE `test_acua_data`.`factors` SET `FI_Importance` = %s WHERE (`factors_id` = %s); "%(k,j)
               mycursor.execute(queryFI)
               my_conn.commit()
            for i,j in OPMN_vals:
               queryFI="UPDATE `test_acua_data`.`factors` SET `OPMN_General_Average_Importance` = %s WHERE (`factors_id` = %s); "%(j,i)
               mycursor.execute(queryFI)
               my_conn.commit()
            Model_ID_query=""" SELECT 
             *
           FROM
             model_id"""
         
         
            model_id_tab=pd.read_sql(Model_ID_query,con=engine)
            
    
            ID_model_list2=list(model_id_tab.iloc[2:6,1])
            Saving_choose.set_menu(ID_model_list2[0] ,*ID_model_list2)
            ID_model_list3=list(model_id_tab.iloc[1:6,1])
            Model_choose.set_menu(ID_model_list3[saved_model-1] ,*ID_model_list3)
            Checkgui.destroy()

        def no_command():
            Checkgui.destroy()
            
        yesButton=ttk.Button(Checkgui,text='Yes, I want to save this new model', command= yes_command)
        yesButton.pack(pady=10,padx=20)
        noButton=ttk.Button(Checkgui,text='I do not want to save this model',command=no_command)
        noButton.pack(pady=10,padx=20)
        


def reset_model():
    global RP_model, FI_model,Pipe_DB_All,changingEntry
    user='root' #user name for the user that is connecting to the database
    password='password' #password for the user that is connecting to the database
    host='localhost' #host name or ip address, for local machines it is localhost otherwise it is the server ip 
    database='test_acua_data' #database which you are connecting to
    
    my_conn=mysql.connector.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database)
    
        
    mycursor=my_conn.cursor()
    
    Model_ID_query=""" SELECT 
      *
    FROM
      model_id"""
    model_id_tab=pd.read_sql(Model_ID_query,con=engine)
    
    
    
    desired_model=Model_ID.get()
    desired_df=model_id_tab.loc[model_id_tab["Model_name"]==desired_model,['Model_ID']]
    desired_index=desired_df.iloc[0,0]    
    
    
    Pipe_DB_All['Random Forest RP']=0
    Pipe_DB_All['Random Forest FI']=0
    Pipe_DB_All['Random Forest OPMN']=0
    
    factor_query="""SELECT * From factors WHERE Model_ID=%s""" %desired_index
    
    base_fac=pd.read_sql(factor_query,con=engine)
    RP_fact=base_fac[base_fac['RP_Importance']>0]
    RP_factors=RP_fact["Factors"]
    factor_labels=list(RP_fact["Factor_label"])
    RP_dependentvariable='True_Risk_Probability'

    all_RP_data=Pipe_DB_All.loc[:,RP_factors]
   
    
    FI_fact=base_fac[base_fac['FI_Importance']>0]
    fi_factors=FI_fact["Factors"]
    fi_factor_labels=list(FI_fact["Factor_label"])
    fi_dependentvariable='True_Failure_Impact'

    all_FI_data=Pipe_DB_All.loc[:,fi_factors]  
    if RP_factors.empty== True:
        Checkgui=tk.Toplevel()
        Checkgui.title("Error!")
        errorLabel=ttk.Label(Checkgui,text='Make sure you select a model that is not empty!')
        errorLabel.pack(pady=10,padx=10)
    else:
        Pipe_DB_All['Random Forest RP']=0
        Pipe_DB_All['Random Forest FI']=0
        Pipe_DB_All['Random Forest OPMN']=0
        del RP_model, FI_model
        RP_model=make_model(pipe_DB,factors=RP_factors,dependentvariable=RP_dependentvariable,labels=factor_labels)
        FI_model=make_model(pipe_DB,factors=fi_factors,dependentvariable=fi_dependentvariable,labels=fi_factor_labels)  
        
        RP_prediction=RP_model.model.predict(all_RP_data)
        FI_prediction=FI_model.model.predict(all_FI_data)
        
        OPMN_vals=RP_prediction*FI_prediction
        
        
        Pipe_DB_All['Random Forest RP']=RP_prediction
        Pipe_DB_All['Random Forest FI']=FI_prediction
        Pipe_DB_All['Random Forest OPMN']=OPMN_vals
        
        
        reset_query="UPDATE `test_acua_data`.`model_id` SET `Model_name` = '%s' WHERE (`Model_ID` = '0');"%str(desired_index)
        mycursor.execute(reset_query)
        my_conn.commit()
  
        top_filter(10,Pipe_DB_All)
        search_df()
        
        changingEntry.config(state="normal")
        changingEntry.delete(0,'end')
        changingEntry.insert(0,desired_model)
        changingEntry.config(state="disabled")
        
        #trvwSearch.delete(*trvwSearch.get_children())
        for i in [trvwNewRP,Base_trvwRP,Base_trvwFI,trvwNewFI]:
            i.delete(*i.get_children())
        
        NewRP_Data=base_fac[base_fac['RP_Importance']>0].loc[:,['Factors','Factor_label']]
        show_factors(trvwNewRP,NewRP_Data)
        
        BaseRP_Data=base_fac[base_fac['RP_Importance']>0].loc[:,['Factors','Factor_label']]
        show_factors(Base_trvwRP,BaseRP_Data)
        
        BaseFI_Data=base_fac[base_fac['FI_Importance']>0].loc[:,['Factors','Factor_label']]
        show_factors(Base_trvwFI,BaseFI_Data)
        
        NewFI_Data=base_fac[base_fac['FI_Importance']>0].loc[:,['Factors','Factor_label']]
        show_factors(trvwNewFI,NewFI_Data)
        
        for i in [Base_RP_acc_entry,Base_FI_acc_entry,New_RP_acc_entry,New_FI_acc_entry]:
            i.config(state="normal")
            i.delete(0,'end')
        
        
        Base_RP_acc_entry.insert(0,"{:.2f}".format(RP_model.accuracy_perc))
        Base_FI_acc_entry.insert(0,"{:.2f}".format(FI_model.accuracy_perc))        
        
        New_RP_acc_entry.insert(0,"{:.2f}".format(RP_model.accuracy_perc))
        New_FI_acc_entry.insert(0,"{:.2f}".format(FI_model.accuracy_perc))
        
        for i in [Base_RP_acc_entry,Base_FI_acc_entry,New_RP_acc_entry,New_FI_acc_entry]:
            i.config(state="disable")        
        

        Dropselection=dropModel._variable.get()
        if Dropselection in 'Confusion Matrix':
            for i in [BaseRP_Frame,BaseFI_Frame,NewRP_Frame,NewFI_Frame]:
                for child in i.winfo_children():
                    child.destroy()    
            Base_RPCanvas = FigureCanvasTkAgg(RP_model.conf_matrix,master=BaseRP_Frame )
            Base_RPCanvas.draw()
            Base_RPCanvas.get_tk_widget().pack()
            
            Base_FICanvas = FigureCanvasTkAgg(FI_model.conf_matrix,master=BaseFI_Frame )
            Base_FICanvas.draw()
            Base_FICanvas.get_tk_widget().pack()
            
            NEW_RPCanvas = FigureCanvasTkAgg(RP_model.conf_matrix,master=NewRP_Frame )
            NEW_RPCanvas.draw()
            NEW_RPCanvas.get_tk_widget().pack()
            
            NEW_FICanvas = FigureCanvasTkAgg(FI_model.conf_matrix,master=NewFI_Frame )
            NEW_FICanvas.draw()
            NEW_FICanvas.get_tk_widget().pack()         
        elif Dropselection in 'Feature Importance':
            for i in [BaseRP_Frame,BaseFI_Frame,NewRP_Frame,NewFI_Frame]:
                for child in i.winfo_children():
                    child.destroy()
                
            Base_RPCanvas = FigureCanvasTkAgg(RP_model.plot,master=BaseRP_Frame )
            Base_RPCanvas.draw()
            Base_RPCanvas.get_tk_widget().pack()
            
            Base_FICanvas = FigureCanvasTkAgg(FI_model.plot,master=BaseFI_Frame )
            Base_FICanvas.draw()
            Base_FICanvas.get_tk_widget().pack()
            
            NEW_RPCanvas = FigureCanvasTkAgg(RP_model.plot,master=NewRP_Frame )
            NEW_RPCanvas.draw()
            NEW_RPCanvas.get_tk_widget().pack()
            
            NEW_FICanvas = FigureCanvasTkAgg(FI_model.plot,master=NewFI_Frame )
            NEW_FICanvas.draw()
            NEW_FICanvas.get_tk_widget().pack()                        
            


##Frames for model selection and saving

savingFrame=ttk.Frame(ModelBuild,style='BLU.TFrame')
savingFrame.grid(row=3,column=2,columnspan=2)

savingLabel=ttk.Label(savingFrame,text='Which Slot would you like to save the Model to?',style='BLU.TLabel')
savingLabel.grid(row=0,column=0)


Saving_choose=ttk.OptionMenu(savingFrame, Save_ID, ID_model_list2[0],*ID_model_list2)
Saving_choose.config(width=27)
Saving_choose.grid(row=0,column=1)

savingLabel2=ttk.Label(savingFrame,text='Enter Name',style='BLU.TLabel')
savingLabel2.grid(row=1,column=0)

SavingEntry=ttk.Entry(savingFrame)
SavingEntry.grid(row=1,column=1)
SavingEntry.config(width=32)

SaveNewModel=ttk.Button(savingFrame,text='Save New Models',command=save_new_model)
SaveNewModel.grid(row=2,column=0,columnspan=2)




changingFrame=ttk.Frame(ModelBuild,style='BLU.TFrame')
changingFrame.grid(row=3,column=0,columnspan=2)

changingLabel=ttk.Label(changingFrame,text='Current Model',style='BLU.TLabel')
changingLabel.grid(row=0,column=0)


Model_choose=ttk.OptionMenu(changingFrame, Model_ID, ID_model_list[saved_model-1],*ID_model_list)
Model_choose.config(width=27)
Model_choose.grid(row=1,column=1)

changingEntry=ttk.Entry(changingFrame)
changingEntry.grid(row=0,column=1)
changingEntry.insert(0,ID_model_list[saved_model-1])
changingEntry.config(width=32,state="disabled")

changingModel=ttk.Button(changingFrame,text='Change Base model',command=lambda: reset_model())
changingModel.grid(row=1,column=0)



# #drop down menu for different graphs
# #https://stackoverflow.com/questions/25216135/more-on-tkinter-optionmenu-first-option-vanishes

ModelOption= ['Feature Importance', 'Confusion Matrix']
SelectS2= StringVar()
dropModel=ttk.OptionMenu(ModelBuild,SelectS2, 'Feature Importance',
                         *ModelOption,
                         command=change_graph
                        )
dropModel.config(width=20)
dropModel.grid(row=0,column=1)

# #Base Risk Probability frame
BaseRP_Frame= ttk.LabelFrame(ModelBuild, text="Base - Risk Probability")
BaseRP_Frame.grid(row=1,column=1, sticky='ns')

Base_RPCanvas = FigureCanvasTkAgg(RP_model.plot,master=BaseRP_Frame)
Base_RPCanvas.draw()
Base_RPCanvas.get_tk_widget().pack()

#Base Failure impact Picture frame
BaseFI_Frame= ttk.LabelFrame(ModelBuild, text="Base - Failure Impact")
BaseFI_Frame.grid(row=2,column=1, sticky='ns')

Base_FICanvas = FigureCanvasTkAgg(FI_model.plot,master=BaseFI_Frame )
Base_FICanvas.draw()
Base_FICanvas.get_tk_widget().pack()

#New Risk Probability frame
NewRP_Frame= ttk.LabelFrame(ModelBuild, text="New - Risk Probability",width=436)
NewRP_Frame.grid(row=1,column=3, sticky='ns')

NEW_RPCanvas = FigureCanvasTkAgg(RP_model.plot,master=NewRP_Frame )
NEW_RPCanvas.draw()
NEW_RPCanvas.get_tk_widget().pack()


#Base Failure impact Picture frame
NewFI_Frame= ttk.LabelFrame(ModelBuild, text="New - Failure Impact",width=436)
NewFI_Frame.grid(row=2,column=3, sticky='ns')

NEW_FICanvas = FigureCanvasTkAgg(FI_model.plot,master=NewFI_Frame )
NEW_FICanvas.draw()
NEW_FICanvas.get_tk_widget().pack()


# ###############Frame for displaying Base Variables FI and RP##################
# ##########                   RP                   ##################


def show_factors(Treeview,dataframe):
    
    for column in Treeview["columns"]:
        Treeview.heading(column, text=column)# let the column heading = column name

    dataframe_rows = dataframe['Factor_label'].to_numpy().tolist() # turns the dataframe into a list of lists
    for row in dataframe_rows:
        Treeview.insert("", "end", values=(row,))

    
Base_RP_Variables=ttk.Frame(ModelBuild)
Base_RP_Variables.grid(row=1,column=0)

Base_RP_Label=ttk.Label(Base_RP_Variables,text='Base Risk Probability Model')
Base_RP_Label.grid(row=0,column=0, columnspan=2)

Base_RP_treeview=ttk.Frame(Base_RP_Variables,height=1000,width=100)
Base_RP_treeview.grid(row=3,column=0,columnspan=2,sticky='ew')

BaseRP_Data=base_fac[base_fac['RP_Importance']>0].loc[:,['Factors','Factor_label']]
Variable_Col=list(['Factors'])

Base_trvwRP= ttk.Treeview(Base_RP_treeview,columns=Variable_Col,show='headings')
Base_trvwRP.pack(padx=0, pady=0, fill='both', expand=True,side='left')
show_factors(Base_trvwRP,BaseRP_Data)

Base_RP_scrollbary=ttk.Scrollbar(Base_RP_treeview, orient="vertical", command=Base_trvwRP.yview) 
Base_RP_scrollbary.pack(side="right", fill="y") 
Base_trvwRP.configure(yscrollcommand=Base_RP_scrollbary.set) 


Base_RP_acc_Label=ttk.Label(Base_RP_Variables,text='Accuracy %:')
Base_RP_acc_Label.grid(row=4,column=0)


Base_RP_acc_entry=ttk.Entry(Base_RP_Variables)
Base_RP_acc_entry.grid(row=4,column=1)
Base_RP_acc_entry.insert(0,"{:.2f}".format(RP_model.accuracy_perc))
Base_RP_acc_entry.config(state='disabled')


# ##########                  FI                   ##################
Base_FI_Variables=ttk.Frame(ModelBuild)
Base_FI_Variables.grid(row=2,column=0)


Base_FI_Label=ttk.Label(Base_FI_Variables,text='Base Failure Impact Model')
Base_FI_Label.grid(row=0,column=0, columnspan=2)

Base_FI_treeview=ttk.Frame(Base_FI_Variables,height=1000,width=100)
Base_FI_treeview.grid(row=3,column=0,columnspan=2,sticky='ew')

BaseFI_Data=base_fac[base_fac['FI_Importance']>0].loc[:,['Factors','Factor_label']]
Variable_Col=list(['Factors'])

Base_trvwFI= ttk.Treeview(Base_FI_treeview,columns=Variable_Col,show='headings')
Base_trvwFI.pack(padx=0, pady=0, fill='both', expand=True,side='left')
show_factors(Base_trvwFI,BaseFI_Data)

Base_FI_scrollbary=ttk.Scrollbar(Base_FI_treeview, orient="vertical", command=Base_trvwFI.yview) 
Base_FI_scrollbary.pack(side="right", fill="y") 
Base_trvwFI.configure(yscrollcommand=Base_FI_scrollbary.set) 


Base_FI_acc_Label=ttk.Label(Base_FI_Variables,text='Accuracy %:')
Base_FI_acc_Label.grid(row=4,column=0)

Base_FI_acc_entry=ttk.Entry(Base_FI_Variables)
Base_FI_acc_entry.grid(row=4,column=1)
Base_FI_acc_entry.insert(0,"{:.2f}".format(FI_model.accuracy_perc))
Base_FI_acc_entry.config(state='disabled')



# ###################Frame for new variables#########################
# ######### RP ###############
RP_variables=ttk.Frame(ModelBuild)
RP_variables.grid(row=1,column=2)
# #how to expand a widget using .grid
# #https://stackoverflow.com/questions/52472105/does-grid-has-a-fill-attribute-like-pack-in-tkinter


NEW_RP_Label=ttk.Label(RP_variables,text='New Risk Probability Model')
NEW_RP_Label.grid(row=0,column=0, columnspan=2)


FeatOption=base_fac.loc[:,'Factor_label']
SelectFeatRP= StringVar()
dropFeatRP=ttk.OptionMenu(RP_variables,SelectFeatRP, FeatOption[0],*FeatOption)
dropFeatRP.config(width=30)
dropFeatRP.grid(row=1,column=0,columnspan=2,sticky='ew')

#lambda allows you to use functions with arguments in the command line
RPInsertButton=ttk.Button(RP_variables,text='Insert Variable',width=15,command=lambda: insert_var(dropFeatRP,trvwNewRP))
RPInsertButton.grid(row=2,column=0,sticky='ew')

RPInsertButton=ttk.Button(RP_variables,text='Delete Variable',width=15,command=lambda: delete_record(trvwNewRP))
RPInsertButton.grid(row=2,column=1,sticky='ew')

RP_treeview=ttk.Frame(RP_variables,height=800,width=100)
RP_treeview.grid(row=3,column=0,columnspan=2,sticky='ew')


NewRP_Data=base_fac[base_fac['RP_Importance']>0].loc[:,['Factors','Factor_label']]
trvwNewRP= ttk.Treeview(RP_treeview,columns="Factor",show='headings')
trvwNewRP.pack(padx=0, pady=0, fill='both', expand=True,side='left')
show_factors(trvwNewRP,NewRP_Data)

NEWRP_scrollbary=ttk.Scrollbar(RP_treeview, orient="vertical", command=trvwNewRP.yview) 
NEWRP_scrollbary.pack(side="right", fill="y") 
trvwNewRP.configure(yscrollcommand=NEWRP_scrollbary.set) 

New_RP_acc_Label=ttk.Label(RP_variables,text='Accuracy %:')
New_RP_acc_Label.grid(row=4,column=0)

New_RP_acc_entry=ttk.Entry(RP_variables)
New_RP_acc_entry.grid(row=4,column=1)
New_RP_acc_entry.insert(0,"{:.2f}".format(RP_model.accuracy_perc))
New_RP_acc_entry.config(state='disabled')

RPNewModelButton=ttk.Button(RP_variables,text='Run New Model'
                            ,command=lambda:run_new_model(trvwNewRP,NewRP_Frame,dropModel,model='RP')
                           )
RPNewModelButton.grid(row=5,column=0,columnspan=2,sticky='ew')

# ####################### FI ##########
FI_variables=ttk.Frame(ModelBuild)
FI_variables.grid(row=2,column=2)
#how to expand a widget using .grid
#https://stackoverflow.com/questions/52472105/does-grid-has-a-fill-attribute-like-pack-in-tkinter


NEW_FI_Label=ttk.Label(FI_variables,text='New Failure Impact Model')
NEW_FI_Label.grid(row=0,column=0, columnspan=2)

SelectFeatFI= StringVar()
dropFeatFI=ttk.OptionMenu(FI_variables,SelectFeatFI, FeatOption[0],*FeatOption)
dropFeatFI.config(width=30)
dropFeatFI.grid(row=1,column=0,columnspan=2)

#lambda allows you to use functions with arguments in the command line
FIInsertButton=ttk.Button(FI_variables,text='Insert Variable',width=15,command=lambda: insert_var(dropFeatFI,trvwNewFI))
FIInsertButton.grid(row=2,column=0,sticky='ew')

FIInsertButton=ttk.Button(FI_variables,text='Delete Variable',width=15,command=lambda: delete_record(trvwNewFI))
FIInsertButton.grid(row=2,column=1,sticky='ew')

FI_treeview=ttk.Frame(FI_variables,height=800,width=100)
FI_treeview.grid(row=3,column=0,columnspan=2,sticky='ew')

NewFI_Data=base_fac[base_fac['FI_Importance']>0].loc[:,['Factors','Factor_label']]
Variable_Col=list(['Factors'])

trvwNewFI= ttk.Treeview(FI_treeview,columns=Variable_Col,show='headings')
trvwNewFI.pack(padx=0, pady=0, fill='both', expand=True,side='left')
show_factors(trvwNewFI,NewFI_Data)

NEWFI_scrollbary=ttk.Scrollbar(FI_treeview, orient="vertical", command=trvwNewFI.yview) 
NEWFI_scrollbary.pack(side="right", fill="y") 
trvwNewFI.configure(yscrollcommand=NEWFI_scrollbary.set) 


New_FI_acc_Label=ttk.Label(FI_variables,text='Accuracy %:')
New_FI_acc_Label.grid(row=4,column=0)

New_FI_acc_entry=ttk.Entry(FI_variables)
New_FI_acc_entry.grid(row=4,column=1)
New_FI_acc_entry.insert(0,"{:.2f}".format(FI_model.accuracy_perc))
New_FI_acc_entry.config(state='disabled')

FINewModelButton=ttk.Button(FI_variables,text='Run New Model'
                            ,command=lambda:run_new_model(trvwNewFI,NewFI_Frame,dropModel,model='FI'))
FINewModelButton.grid(row=5,column=0,columnspan=2,sticky='ew')




# =============================================================================
#             run_new_model(trvwNewRP,NewRP_Frame,dropModel,model='RP')
#             run_new_model(trvwNewFI,NewFI_Frame,dropModel,model='FI')
#             
# =============================================================================
        
    #print(RP_prediction)
    #print(FI_prediction)
    #print(OPMN_vals)




# =============================================================================
# NewModelButton=ttk.Button(ModelBuild,text='test button',
# # =============================================================================
# #                           text='Run New Models',
# #                          command=run_both_new_model
# # =============================================================================
#                             command=lambda: reset_model()
#                          )
# NewModelButton.grid(row=3,column=3)
# =============================================================================


ACUAgui.mainloop()