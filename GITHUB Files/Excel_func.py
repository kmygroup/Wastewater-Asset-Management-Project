# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:19:36 2022

@author: jpste
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd



import xlsxwriter
from xlsxwriter.utility import xl_range
import os
from sqlalchemy import create_engine


def get_col_widths(dataframe):
    # First we find the maximum length of the index column   
    idx_max = max([len(str(s)) for s in dataframe.index.values] + [len(str(dataframe.index.name))])+2
    # Then, we concatenate this to the max of the lengths of column name and its values for each column, left to right
    return [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [len(col)]) for col in dataframe.columns]

## add in the right dataframe to this function
def make_report(Model_ID,pipedatabase):
    try:
        
        user='root' #user name for the user that is connecting to the database
        password='password' #password for the user that is connecting to the database
        host='localhost' #host name or ip address, for local machines it is localhost otherwise it is the server ip 
        database='test_acua_data' #database which you are connecting to

        engine = create_engine("mysql+pymysql://%s:%s@%s/%s" %(user,
                                                               password,
                                                               host,
                                                               database))
        
        query1=""" SELECT 
                     *
                   FROM
                     pipeline_database"""
                     
                     
        #this is what is giving error
        Pipeline_data=pipedatabase
        #pd.read_sql(query1,con=engine)
        Pipeline_data_columns=list(Pipeline_data.columns)
        relv_columns=[e for e in Pipeline_data_columns if e not in 
                                                   ('Town_ID',"Upstream_Pumpstation_ID",'Location', 'Population',
                                                    "Area_sqmi","Pop_Density","Source", "Recorded_date","Reported_pop/sqmi",
                                                    "Date","Flow_Orientation","Flow_Numerical_Category","Material",
                                                    "Mat_Numerical_Category","Pump_Sat_ID","G_F")]
        Pipeline_data=Pipeline_data.loc[:,relv_columns]
        
        stats=Pipeline_data.describe()
        
        Model_ID_query=""" SELECT 
                 *
               FROM
                 model_id"""
                 
        model_id_tab=pd.read_sql(Model_ID_query,con=engine)
        
        
        save_loc=Model_ID.get()    
        db_sel=model_id_tab[model_id_tab['Model_name'].isin([save_loc])]
        save_sel=str(db_sel.iloc[0,0])
        
        #change the columns to be the ones we want
        OPMN_desc=stats
        #Random Forest OPMN
        #OPMN
        OPMN_ranking=pd.DataFrame(Pipeline_data['Random Forest OPMN'].value_counts(sort=False))
        OPMN_ranking['new column']=0
        for i in OPMN_ranking.index:
            string="Assets with OPMN of %s:"%i
            OPMN_ranking.loc[i,'new column']=string

        #Change this dataframe when the features are uploaded to the database
        query2=""" SELECT 
                     Factor_label, RP_Importance,FI_Importance,OPMN_General_Average_Importance
                   FROM
                     factors WHERE Model_ID=%s"""%save_sel
        Features=pd.read_sql(query2,con=engine)
            
        excel_file='Report_ACUA.xlsx'
        ###SHEET Names########
        Sheet1= "Model Info"
        Sheet2="Model Statistics"
        Sheet3="Pipeline Database"
        Sheet4='data sheet'
        ######################


        writer=pd.ExcelWriter(excel_file, engine='xlsxwriter')
        # Access the Pandas xlsxwriter Excel file.
        workbook = writer.book


        Features.to_excel(writer, sheet_name=Sheet1, startcol= 3,startrow=2)
        worksheet = writer.sheets[Sheet1]
        OPMN_ranking.to_excel(writer,sheet_name=Sheet4)

        worksheet2 = writer.sheets[Sheet4]
        Features.to_excel(writer, sheet_name=Sheet4, startcol= 6,startrow=2)

        #uncomment the hide when all said and done
        worksheet2.hide()
        #####FORMATS##########
        formatRed=workbook.add_format({"bg_color": "#FF5353"})
        formatYel=workbook.add_format({"bg_color": "#FFF467"})
        formatGre=workbook.add_format({"bg_color": "#5CDC6B"})
        HeaderFormat=workbook.add_format({'bold':True,
                                         'underline': True,
                                         'bg_color':'#D9D9D9',
                                         'align':'center'})
        TableFormat=workbook.add_format({'bold':True,
                                         #'underline': True,
                                         'border': 1,
                                         #'bg_color':'#D9D9D9',
                                         'align':'center'})
        formatPerc=workbook.add_format({'num_format': '0.00%'})

        OPMN_desc.to_excel(writer,sheet_name=Sheet2)
        worksheet3=writer.sheets[Sheet2]
        for i, width in enumerate(get_col_widths(OPMN_desc)):
            worksheet3.set_column(i, i, width)
       # for column in OPMN_desc:
          #  print(OPMN_desc[column].astype(str).max())
            #column_width=max(OPMN_desc[column].astype(str).map(len).max(), len(column))
            #column_width=max(float(OPMN_desc[column].astype(str).max()), len(column))
            #col_idx = OPMN_desc.columns.get_loc(column)
        #if you do not add 1 to the start and end col index then the index will be counted in the adjustment
        #which thorws off the widths
            #worksheet3.set_column(col_idx+1,col_idx+1,column_width)


        Pipeline_data.to_excel(writer,sheet_name=Sheet3)
        worksheet4=writer.sheets[Sheet3]
        for i, width in enumerate(get_col_widths(Pipeline_data)):
            worksheet4.set_column(i, i, width)        
        #for column in Pipeline_data:
           # print(Pipeline_data[column].astype(str).max()+2)
            #column_width=max(float(Pipeline_data[column].astype(str).max()+2), len(column))
            #column_width=max(Pipeline_data[column].astype(str).map(len).max()+2, len(column))
            #col_idx = Pipeline_data.columns.get_loc(column)
            #if you do not add 1 to the start and end col index then the index will be counted in the adjustment
            #which thorws off the widths
            #worksheet4.set_column(col_idx+1,col_idx+1,column_width)

        max_row=len(Pipeline_data)    
        column_no=Pipeline_data.columns.get_loc("Random Forest OPMN")

    #xl_range returns a range in an excel format ie: A1:B22
    #find the location of OPMN column and add 1 to column to adjust for index

        OPMN_range=xl_range(1,column_no+1,max_row,column_no+1)    
        worksheet.write("A1","Number value of OPMN",HeaderFormat)
        worksheet.set_column(0,0,25)
        worksheet.write("B1", "Amount of assets at OPMN Rank",HeaderFormat)
        worksheet.set_column(1,1,30)
        #set column width of index
        #worksheet.set_column(3,3,5)
        worksheet.set_column(4,4,25)
        worksheet.write('F1','RP Score',HeaderFormat)
        worksheet.write('F2',5)
        worksheet.set_column(5,5,15)
        #worksheet.set_column(4,4,15)
        worksheet.write('G1','FI Score',HeaderFormat)
        worksheet.write('G2',5)
        worksheet.set_column(6,6,15)
        worksheet.set_column(7,7,35)
        worksheet.write('I3','OPMN Score Weighted Average', TableFormat)
        worksheet.set_column(8,8,35)
        worksheet.write('D3','Feature ID', TableFormat)
        worksheet.set_column(3,3,10)
        
        OPMN_scores=np.linspace(1,25,25,dtype=int)
        Label_range=1
        Value_range=1
        OPMN_val=1
        for i in OPMN_scores:
            Label_placement=xl_range(Label_range,0,Label_range,0)
            Value_placement=xl_range(Value_range,1,Value_range,1)
            worksheet.write('%s'%Label_placement, "Assets with OPMN of %s:"% OPMN_val);
            worksheet.write_formula('%s'%Value_placement,"=COUNTIF('%s'!%s,%s)" % (Sheet3,OPMN_range,OPMN_val))
            Label_range=Label_range+1
            Value_range=Value_range+1
            OPMN_val=OPMN_val+1


        worksheet2.write_formula('H2',"'%s'!F2"%Sheet1)  
        worksheet2.write_formula('I2',"'%s'!G2"%Sheet1)
        row_num=4
        for i in range(len(Features)):    
            worksheet2.write_formula('L%s'%row_num,'==((H2/5)*I%s+(I2/5)*J%s)/2'%(row_num,row_num))
            row_num=row_num+1

        row_num=4    
        for i in range(len(Features)):
            worksheet.write_formula("I%s"%row_num,"'%s'!L%s/SUM('%s'!$L$4:$L$13)"%(Sheet4,row_num,Sheet4))
            row_num=row_num+1  

        worksheet.conditional_format('F4:I13',{'type': 'formula',
                                                  'criteria': "='%s'!A1>-1"%Sheet4,
                                                  'format': formatPerc})

        OPMN_pie=workbook.add_chart({'type':'pie'})
        #https://xlsxwriter.readthedocs.io/chart.html
        OPMN_pie.add_series({
             'categories': "='%s'!C2:C12"%Sheet4,
             'values':     "='%s'!B2:B12"%Sheet4,
             'data_labels': {'category': True, 'percentage': True, 'leader_lines': True}
            })

        OPMN_pie.set_legend({'none': True})
        OPMN_pie.set_title({'name': 'OPMN Breakdown'})
        worksheet.insert_chart('C14', OPMN_pie,
                           {'x_offset': 15,'y_offset': 10,'x_scale':1.5,'y_scale':1.5})

        Gen_Avg_pie=workbook.add_chart({'type':'pie'})
        #https://xlsxwriter.readthedocs.io/chart.html
        Gen_Avg_pie.add_series({
             'categories': "='%s'!E4:E13"%Sheet1,
             'values':     "='%s'!H4:H13"%Sheet1,
             'data_labels': {'category': True, 'percentage': True, 'leader_lines': True}
            })
        Gen_Avg_pie.set_legend({'none': True})
        Gen_Avg_pie.set_title({'name': 'General Average Importance'})
        worksheet.insert_chart('J2', Gen_Avg_pie,
                           {'x_offset': 30,'y_offset': 10,'x_scale':1.35,'y_scale':1.35})


        Weg_Avg_pie=workbook.add_chart({'type':'pie'})
        #https://xlsxwriter.readthedocs.io/chart.html
        Weg_Avg_pie.add_series({
             'categories': "='%s'!E4:E13"%Sheet1,
             'values':     "='%s'!I4:I13"%Sheet1,
             'data_labels': {'category': True, 'percentage': True, 'leader_lines': True}  
            })
        Weg_Avg_pie.set_legend({'none': True})
        Weg_Avg_pie.set_title({'name': 'Weighted Average Importance'})
        worksheet.insert_chart('J20', Weg_Avg_pie,
                           {'x_offset': 30,'y_offset': 10,'x_scale':1.35,'y_scale':1.35})


        OPMN_Letter=xlsxwriter.utility.xl_col_to_name(column_no+1)
        #freeze pane on top row
        worksheet4.freeze_panes(1,0)
        Cond_form_range=xl_range(1,0,max_row,len(Pipeline_data.columns))
        worksheet4.conditional_format(Cond_form_range,{'type': 'formula',
                                                      'criteria': '=$%s2>14'%OPMN_Letter,
                                                      'format': formatRed})
        worksheet4.conditional_format(Cond_form_range,{'type': 'formula',
                                                      'criteria': '=$%s2>7'%OPMN_Letter,
                                                      'format': formatYel})
        worksheet4.conditional_format(Cond_form_range,{'type': 'formula',
                                                      'criteria': '=$%s2>0'%OPMN_Letter,
                                                      'format': formatGre})
        writer.save()
        path="%s\\%s" % (os.getcwd(),excel_file )  
        os.startfile(path)
        

    except Exception as e :
        Errorgui = tk.Toplevel()
        Errorgui.title("Error!")
        Style=ttk.Style()
        Style.configure('Error.TLabel',foreground='red')
        w = 350 # width for the Tk root
        h = 50 # height for the Tk root
        # get screen width and height
        ws = Errorgui.winfo_screenwidth() # width of the screen
        hs = Errorgui.winfo_screenheight() # height of the screen
        # calculate x and y coordinates for the Tk root window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        #set the dimensions of the screen  and where it is placed
        Errorgui.geometry('%dx%d+%d+%d' % (w, h, x, y-25))     
        label=ttk.Label(Errorgui,text="Please close report to open a new one%s"%e, style ='Error.TLabel')
        #label=ttk.Label(Errorgui,text="%s" %e, style ='Error.TLabel')
        #label.configure(style='Error.TLabel')
        label.pack(pady=10,padx=10)
        
