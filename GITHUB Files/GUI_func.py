# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:36:29 2022

@author: jpste
"""
import tkinter as tk

from tkinter import ttk


        
def insert_var(dropdown,treeview):
    treeview.insert("", "end", values=[dropdown._variable.get(),'1'],tags=('evenrow',))
    
def delete_record(tree):
    try:
        selected_item = tree.selection()[0] ## get selected item
        tree.delete(selected_item)
    except IndexError:
        #Style=ttk.Style()
        #Style.configure('Error.TLabel',foreground='red')
        Errorgui = tk.Tk()
        Errorgui.title("Error!")
        Style=ttk.Style()
        Style.configure('Error.TLabel',foreground='red')
        w = 250 # width for the Tk root
        h = 50 # height for the Tk root
        # get screen width and height
        ws = Errorgui.winfo_screenwidth() # width of the screen
        hs = Errorgui.winfo_screenheight() # height of the screen
        # calculate x and y coordinates for the Tk root window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        #set the dimensions of the screen  and where it is placed
        Errorgui.geometry('%dx%d+%d+%d' % (w, h, x, y-25))     
        label=ttk.Label(Errorgui,text="Erorr: please make a selection to delete",style ='Error.TLabel')
        #label.configure(style='Error.TLabel')
        label.pack(pady=10,padx=10)