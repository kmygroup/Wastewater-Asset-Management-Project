# -*- coding: utf-8 -*-
"""
Created on Fri Mar 4 11:23:15 2022

@author: aboagy26
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
#%% import data
url = 'C:/Users/aboagy26/OneDrive - Rowan University/Emmanuel_Apau_Aboagye/Wastewater Treatment/ACUA Manuscript/Response to reviewer comments/Final_DF_Emmanuel.xlsx'
data_df = pd.read_excel(url) 

#%% Features and Label Data
risk_prob_features = ['Flow_Type', 'Pipe_Type', 'Years_Since_Last_Inpsection', 'Remaining_Life', 'Pipe_Size', 'Segment_Length', 'Original_Installation ', 'Population_Density']
fail_imp_features = ['Flow_Type', 'Pipe_Size', 'Years_Since_Last_Inpsection', 'Population_Density', 'Flow_Rate', 'Up_or_Downstream']

risk_prob_label = ['Risk_Probability']
fail_imp_label = ['Failure_Impact']
#%%

# for i in range(len(risk_prob_features)):
#     plt.figure(dpi=300)
#     plt.hist(data_df[risk_prob_features[i]], bins=10)
#     plt.title(risk_prob_features[i])
#     plt.show()

#%% Define some functions 
    
def make_model_rf(data, features, label, random_state_=20, train_size_=0.70, Tree_num=27, max_depth_=6):   
    Independent = data.loc[:,features]
    Dependent = np.array(data.loc[:,label])
    Tree_num=Tree_num
    X_train, X_test, Y_train, Y_test = train_test_split(Independent, 
                                                        Dependent,
                                                        train_size=train_size_, 
                                                        random_state=random_state_,
                                                        stratify=Dependent)      
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test  = scalar.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=Tree_num, random_state=random_state_, max_depth=max_depth_)
    model.fit(X_train, Y_train.ravel())
    
    y_pred = model.predict(X_test) 
    Accuracy = accuracy_score(Y_test.ravel(), y_pred)
    Accuracy = float(Accuracy)*100
   
    return {
            'model': model, 
            'Accuracy': Accuracy,
            'X_test': X_test, 
            'Y_test': Y_test, 
            'X_train': X_train, 
            'Y_train': Y_train,
            }

def make_model_xgb(data, features, label, random_state_=20, train_size_=0.70, Tree_num=27, max_depth_=6):   
    Independent = data.loc[:,features]
    Dependent = np.array(data.loc[:,label])
    Tree_num=Tree_num
    X_train, X_test, Y_train, Y_test = train_test_split(Independent, 
                                                        Dependent,
                                                        train_size=train_size_, 
                                                        random_state=random_state_,
                                                        stratify=Dependent)       
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test  = scalar.transform(X_test)  
    
    mapping_array = np.array([0,1,2,3,4])
    model = XGBClassifier(n_estimators=Tree_num, max_depth=max_depth_)
    model.fit(X_train, mapping_array[Y_train-1])
    y_pred = model.predict(X_test) 
    Accuracy = accuracy_score(mapping_array[Y_test-1], y_pred)
    Accuracy = float(Accuracy)*100
    return {
            'model': model, 
            'Accuracy': Accuracy, 
            'X_test': X_test, 
            'Y_test': mapping_array[Y_test-1], 
            'X_train': X_train, 
            'Y_train': mapping_array[Y_train-1],
            }

def plot_feature_importance(importance_rf_RP,
                            names_RP,
                            importance_xg_RP,
                            color_='blue',
                            ):
    
    plt.figure(figsize=(8,18), dpi=700)
    
    plt.subplot(2,1,1)
    feature_names_RP = np.array(names_RP) 
    feature_importance_rf_RP = importance_rf_RP   
    data_rf_RP={'feature_names':feature_names_RP,'feature_importance':feature_importance_rf_RP}
    fi_df_rf_RP = pd.DataFrame(data_rf_RP)   
    fi_df_rf_RP.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    sns.barplot(x=fi_df_rf_RP['feature_importance'], y=fi_df_rf_RP['feature_names'], color=color_)
    
    plt.subplot(2,1,2)
    feature_importance_xg_RP = importance_xg_RP   
    data_xg_RP={'feature_names':feature_names_RP,'feature_importance':feature_importance_xg_RP}
    fi_df_xg_RP = pd.DataFrame(data_xg_RP)   
    fi_df_xg_RP.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    sns.barplot(x=fi_df_xg_RP['feature_importance'], y=fi_df_xg_RP['feature_names'], color=color_)

    plt.show()


#%% Create an instance of the classification models
risk_prob_results_rf = make_model_rf(data_df, risk_prob_features, risk_prob_label, train_size_=0.7)
fail_imp_results_rf = make_model_rf(data_df, fail_imp_features, fail_imp_label, train_size_=0.7)

risk_prob_results_xgb = make_model_xgb(data_df, risk_prob_features, risk_prob_label, train_size_=0.7)
fail_imp_results_xgb = make_model_xgb(data_df, fail_imp_features, fail_imp_label, train_size_=0.7)

#%% Unpack the results
# model_rf_risk_prob = risk_prob_results_rf['model']
# Accuracy_rf_risk_prob = risk_prob_results_rf['Accuracy']
# model_rf_fail_imp = fail_imp_results_rf['model']
# Accuracy_rf_fail_imp = fail_imp_results_rf['Accuracy']

# model_xgb_risk_prob = risk_prob_results_xgb['model']
# Accuracy_xgb_risk_prob = risk_prob_results_xgb['Accuracy']
# model_xgb_fail_imp = fail_imp_results_xgb['model']
# Accuracy_xgb_fail_imp = fail_imp_results_xgb['Accuracy']

X_train_rf_risk_prob = risk_prob_results_rf['X_train']
X_test_rf_risk_prob = risk_prob_results_rf['X_test']
Y_train_rf_risk_prob = risk_prob_results_rf['Y_train']
Y_test_rf_risk_prob = risk_prob_results_rf['Y_test']

X_train_rf_fail_imp = fail_imp_results_rf['X_train']
X_test_rf_fail_imp = fail_imp_results_rf['X_test']
Y_train_rf_fail_imp = fail_imp_results_rf['Y_train']
Y_test_rf_fail_imp = fail_imp_results_rf['Y_test']

X_train_xgb_risk_prob = risk_prob_results_xgb['X_train']
X_test_xgb_risk_prob = risk_prob_results_xgb['X_test']
Y_train_rf_fail_imp = risk_prob_results_xgb['Y_train']
Y_test_xgb_risk_prob = risk_prob_results_xgb['Y_test']

X_train_xgb_fail_imp = fail_imp_results_xgb['X_train']
X_test_xgb_fail_imp = fail_imp_results_xgb['X_test']
Y_train_xgb_fail_imp = fail_imp_results_xgb['Y_train']
Y_test_xgb_fail_imp = fail_imp_results_xgb['Y_test']

#%% Hyperparameter tuning spcifications using 'hyperopt'
space = {
    'n_estimators': hp.choice('n_estimators', range(10,500,100)),
    'max_depth': hp.choice('max_depth', range(1, 21)),
    #'random_state': hp.choice('random_state', range(1, 500)),
    #'min_samples_split': hp.uniform('min_samples_split', 0.1, 1), 
    #'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5), 
}

def objective(params):
    skf = StratifiedKFold(n_splits=3)
    clf = XGBClassifier(n_estimators=int(params['n_estimators']),
                                 max_depth = params['max_depth'],
                                 random_state= 20
     #                            min_samples_split = params['min_samples_split'],
     #                            min_samples_leaf = params['min_samples_leaf'],
        )
    score = cross_val_score(clf, X_train_xgb_fail_imp, Y_train_xgb_fail_imp, cv=skf).mean()  # change the data for the model your are tuning
    return {'loss': -score, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials)

print('best hyperparameters: ', space_eval(space,best))

# splitting_risk_prob_df = pd.DataFrame({'training': splitting, 'risk_prob_acc_rf':risk_prob_acc_rf, 'risk_prob_acc_xg':risk_prob_acc_xg}) 
#%% Refiting model with optimal hyperparameters
risk_prob_results_rf_tuned = make_model_rf(data_df, risk_prob_features, risk_prob_label, train_size_=0.7,Tree_num=310, max_depth_=3) # 93.36441336441337
fail_imp_results_rf_tuned = make_model_rf(data_df, fail_imp_features, fail_imp_label, train_size_=0.7, Tree_num=110, max_depth_=19) # 91.31313131313131

risk_prob_results_xgb_tuned = make_model_xgb(data_df, risk_prob_features, risk_prob_label, train_size_=0.7, Tree_num=10, max_depth_=1) 
fail_imp_results_xgb_tuned = make_model_xgb(data_df, fail_imp_features, fail_imp_label, train_size_=0.7, Tree_num=110, max_depth_=4) 


#%% Plot feature importance
plot_feature_importance(risk_prob_results_rf_tuned['model'].feature_importances_,
                            risk_prob_features,
                            risk_prob_results_xgb_tuned['model'].feature_importances_,
                            color_='blue'
                            )

plot_feature_importance(fail_imp_results_rf_tuned['model'].feature_importances_,
                            fail_imp_features,
                            fail_imp_results_xgb_tuned['model'].feature_importances_,
                            color_='green'
                            )

#%% Model Predictions
Y_pred_rf_RP = risk_prob_results_rf_tuned['model'].predict(X_test_rf_risk_prob)
Y_pred_rf_FI = fail_imp_results_rf_tuned['model'].predict(X_test_rf_fail_imp)

Y_pred_xg_RP = risk_prob_results_xgb_tuned['model'].predict(X_test_xgb_risk_prob)
Y_pred_xg_FI = fail_imp_results_xgb_tuned['model'].predict(X_test_xgb_fail_imp)

#%% Calculating important model metrics
y_probs_rf_RP = risk_prob_results_rf_tuned['model'].predict_proba(X_test_rf_risk_prob)[:, 1]
y_probs_rf_FI = fail_imp_results_rf_tuned['model'].predict_proba(X_test_rf_fail_imp)[:, 1]

y_probs_xg_RP = risk_prob_results_xgb_tuned['model'].predict_proba(X_test_xgb_risk_prob)[:, 1]
y_probs_xg_FI = fail_imp_results_xgb_tuned['model'].predict_proba(X_test_xgb_fail_imp)[:, 1]


# Accuracy-score
acc_rf_RP = accuracy_score(Y_test_rf_risk_prob, Y_pred_rf_RP)
acc_rf_FI = accuracy_score(Y_test_rf_fail_imp, Y_pred_rf_FI)

acc_xg_RP = accuracy_score(Y_test_xgb_risk_prob, Y_pred_xg_RP)
acc_xg_FI = accuracy_score(Y_test_xgb_fail_imp, Y_pred_xg_FI)

# F1-score
f1_rf_RP = f1_score(Y_test_rf_risk_prob, Y_pred_rf_RP, average='weighted')
f1_rf_FI = f1_score(Y_test_rf_fail_imp, Y_pred_rf_FI, average='weighted')

f1_xg_RP = f1_score(Y_test_xgb_risk_prob, Y_pred_xg_RP, average='weighted')
f1_xg_FI = f1_score(Y_test_xgb_fail_imp, Y_pred_xg_FI, average='weighted')

# Precision-score
precision_rf_RP = precision_score(Y_test_rf_risk_prob, Y_pred_rf_RP, average='weighted')
precision_rf_FI = precision_score(Y_test_rf_fail_imp, Y_pred_rf_FI, average='weighted')

precision_xg_RP = precision_score(Y_test_xgb_risk_prob, Y_pred_xg_RP, average='weighted')
precision_xg_FI = precision_score(Y_test_xgb_fail_imp, Y_pred_xg_FI, average='weighted')

# Recall-score
recall_rf_RP = recall_score(Y_test_rf_risk_prob, Y_pred_rf_RP, average='weighted')
recall_rf_FI = recall_score(Y_test_rf_fail_imp, Y_pred_rf_FI, average='weighted')

recall_xg_RP = recall_score(Y_test_xgb_risk_prob, Y_pred_xg_RP, average='weighted')
recall_xg_FI = recall_score(Y_test_xgb_fail_imp, Y_pred_xg_FI, average='weighted')

#%%
f1_scores = [round(f1_rf_RP*100, 1), round(f1_rf_FI*100, 1), round(f1_xg_RP*100, 1), round(f1_xg_FI*100, 1)] 
precision_scores = [round(precision_rf_RP*100, 1), round(precision_rf_FI*100, 1), round(precision_xg_RP*100, 1), round(precision_xg_FI*100, 1)]
recall_scores = [round(recall_rf_RP*100, 1), round(recall_rf_FI*100, 1), round(recall_xg_RP*100, 1), round(recall_xg_FI*100, 1)] 
accuracy_scores = [round(acc_rf_RP*100, 1), round(acc_rf_FI*100, 1), round(acc_xg_RP*100, 1), round(acc_xg_FI*100, 1)] 
f1_scores, precision_scores, recall_scores, accuracy_scores

#%% Risk Factor (OPMN) Calculation
rp_feature_set = StandardScaler().fit_transform(data_df.loc[:,risk_prob_features])
fi_feature_set = StandardScaler().fit_transform(data_df.loc[:,fail_imp_features])

risk_probability_prediction = risk_prob_results_rf_tuned['model'].predict(rp_feature_set)
failure_impact_prediction = fail_imp_results_rf_tuned['model'].predict(fi_feature_set)
risk_factor_prediction = risk_prob_results_rf_tuned['model'].predict(rp_feature_set) * fail_imp_results_rf_tuned['model'].predict(fi_feature_set)
risk_probability_actual = data_df['Risk_Probability']
failure_impact_actual = data_df['Failure_Impact']
risk_factor_actual = risk_probability_actual * failure_impact_actual

#%% Put prediction and actual in a dataframe and export to excel
data_df_results = pd.DataFrame(risk_probability_actual.values, index=data_df['Location'], columns=['Risk_Probability_Actual'])
data_df_results['Risk_Probability_Predicted'] = risk_probability_prediction
data_df_results['Failure_Impact_Actual'] = failure_impact_actual.values
data_df_results['Failure_Impact_Predicted'] = failure_impact_prediction
data_df_results['Risk_Factor_Actual'] = risk_factor_actual.values
data_df_results['Risk_Factor_Predicted'] = risk_factor_prediction
data_df_results.to_excel('C:/Users/aboagy26/OneDrive - Rowan University/Emmanuel_Apau_Aboagye/Wastewater Treatment/ACUA Manuscript/Response to reviewer comments/Results.xlsx')

#%%


