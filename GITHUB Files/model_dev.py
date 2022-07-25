# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:10:12 2022

@author: jpste
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from collections import namedtuple


def diagonal_heatmap(m):
    global matrix
    new_m=np.copy(m)
    np.fill_diagonal(new_m,0)
    
    vmin = np.min(new_m)
    vmax = np.max(new_m)    
    
    
    matrix=plt.figure(num=2,figsize=(6,4))
    sns.heatmap(m, annot=True,vmin=vmin, vmax=vmax,fmt='g')
    diag_nan = np.full_like(m, np.nan, dtype=float)
    np.fill_diagonal(diag_nan, np.diag(m))
    
    my_cmap = plt.get_cmap('viridis')
    my_cmap.set_over('green')
    
    sns.heatmap(diag_nan, annot=True, 
                cmap=my_cmap, fmt='g',
                vmin=-2, vmax=-1,cbar=False,
                xticklabels=[1,2,3,4,5],yticklabels=[1,2,3,4,5])
    plt.xlabel('Predicted')
    plt.ylabel('Truth') 
    return(matrix)


def make_model(data,factors=[""],dependentvariable=[""], labels=[""],
               Random_State=20,training=0.68,Tree_num=27):
    Independent=data.loc[:,factors]

    Dependent=data[dependentvariable].values
    Dependent=Dependent.astype('int')
    Random_State=Random_State
    training=training
    Tree_num=Tree_num
    X_train, X_test, Y_train, Y_test = train_test_split(Independent, 
                                                        Dependent,
                                                        train_size=training, 
                                                        random_state=Random_State)
    
    
    taining_length="the length of the training set is: %d"%len(X_train)
    testing_length="the length of the testing set is: %d"%len(X_test)
    
    model = RandomForestClassifier(n_estimators=Tree_num, random_state=Random_State)
    model.fit(X_train, Y_train)
    prediction = model.predict(Independent) 
    Accuracy=metrics.accuracy_score(Dependent, prediction)
    Accuracy_show=float(Accuracy)*100
    
    feature_list = list(Independent.columns) 
    feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    feature_imp_label=pd.Series(model.feature_importances_, index=list(labels)).sort_values(ascending=False)
    Feature_Importance=pd.DataFrame(feature_imp,columns=['percentage'])
    Feature_Importance.index.name = 'feature'
    Feature_Importance.columns.name = Feature_Importance.index.name
    Feature_Importance.index.name = None
    
    
    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))
        return "{:.1f}%".format(pct, absolute)
    data=feature_imp
    fig=plt.figure(figsize=(6,4));
    plt.pie(data,
            labels=feature_imp_label.index.values,
            autopct=lambda pct: func(pct, data),
            labeldistance=1.15);
    plt.title("Feature Importance");
    plt.savefig("Pie_feature_imp",facecolor="lightblue");
    plt.close(fig=fig)

    All_Depths=[estimator.tree_.max_depth for estimator in model.estimators_]        
    min_depth,max_depth,average_depth=min(All_Depths),max(All_Depths),np.mean(All_Depths)

    DF_index=[]
    for i in range(len(All_Depths)):
        index=str(i+1)
        sentence="Tree "+index+" depth is:"
        DF_index.append(sentence)
    
    #This block makes a dataframe with all the tree depths 
    table_depths=pd.DataFrame(index=DF_index,columns=list('1'))
    table_depths.insert(1,"Depth without primary node",All_Depths)
    table_depths.drop('1',axis=1,inplace=True)

    Cross_val=cross_val_score(model,Independent, Dependent, cv=4)
    Cross_val_score="the scores for all folds are:%s"%Cross_val
    Average_CV="the average score is: %0.3f" %Cross_val.mean()
    Std_dev_CV="the standard deviation is: %0.3f"%Cross_val.std()
    
    
    cm = confusion_matrix(Dependent, prediction)
    diagonal_heatmap(cm)
    plt.close(fig=matrix)
    
    
    #this will create the attribute class for everything to be stored in
    #output=namedtuple('some name', ['name of variables you want'])
    #the tuple is what you want to call the variables outside the function
    Output=namedtuple("a",['X','Y','TrainL','TestL','prediction', 'accuracy','accuracy_perc','feature_importances','plot','min_depth', 
                           'max_depth', 'average_depth','all_depths','cross_validation_scores',
                          'cross_validation_average','Cross_validation_STD','conf_matrix','model'])
    
    
    #the return statement is what the names are inside the function 
    return Output(Independent,Dependent,taining_length,testing_length,prediction,Accuracy,Accuracy_show,Feature_Importance,fig,min_depth,
                  max_depth,average_depth,table_depths,
                  Cross_val_score,Average_CV,Std_dev_CV,matrix,model)