import sys
import os
sys.path.append("../") 
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import pickle
import json
import requests
from exploratory_data_analysis import omnivida_loader as ov
from exploratory_data_analysis import omnivida_util as ovu
import statsmodels.api         as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
from IPython.display import Image, SVG
from sklearn.tree import export_graphviz
import pydotplus
import xgboost as xgb
from xgboost import XGBClassifier

class clinicalModel():
    def __init__(self):
        self.data = self.load_data()
        self.rforest = None
        self.xgboost = None

    def load_data(self):
        adherence, adherence_change = ov.get_adherence_dataset()
        dyspnea = ov.get_dyspnea_dataset()
        act_disag = ov.get_act_disag_dataset()
        act = ov.get_act_dataset()
        vacc = ov.get_vaccination_dataset()

        act_disag.drop(columns=['Unnamed: 0'], axis = 1, inplace=True)
        dyspnea.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)
        act.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)
        vacc.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)

        dyspnea_cp = dyspnea.copy()
        act_disag_cp = act_disag.copy()
        act_cp = act.copy()
        vacc_c = vacc.copy()

        select_fields = ['id_patient', 'survey_date', 'category', 'qualitative_result', 'qualitative_result_change', 'days_since_last_control', 'ongoing_adherence_percentage', 'num_reports']
        adherence_change_analysis = adherence_change[select_fields]

        # Dyspnea dataset
        dyspnea_cp.medical_test = dyspnea_cp.medical_test.astype(int)
        # act_disag dataset
        act_disag_cp.A_ACT = act_disag_cp.A_ACT.astype(int)
        act_disag_cp.B_ACT = act_disag_cp.B_ACT.astype(int)
        act_disag_cp.C_ACT = act_disag_cp.C_ACT.astype(int)
        act_disag_cp.D_ACT = act_disag_cp.D_ACT.astype(int)
        act_disag_cp.FEEDBACK = act_disag_cp.FEEDBACK.astype(int)
        # Act dataset
        act_cp.act_score = act_cp.act_score.astype(int)
        act_cp.result = act_cp.result.astype(int)
        # Vaccination dataset
        select_fields_vacc = ['id_patient', 'emission_date', 'regional_eps', 'service_description', 'diagnostic_eps_description', 'auth_quantity']
        vacc_cp = vacc_c[select_fields_vacc]
        vacc_cp['regional_eps'] = vacc_cp['regional_eps'].astype('category')
        vacc_cp['service_description'] = vacc_cp['service_description'].astype('category')
        vacc_cp['diagnostic_eps_description'] = vacc_cp['diagnostic_eps_description'].astype('category')
        vacc_cp.auth_quantity = vacc_cp.auth_quantity.astype(int)
        adherence_11 = adherence_change_analysis.merge(dyspnea_cp, how='left', on='id_patient')

        # Merging adherence and dyspnea_cp
        merge_adh_dysp = ovu.merge_on_closest_date(df1=adherence_change_analysis, 
                                                df2=dyspnea_cp, 
                                                date_field_df1='survey_date', 
                                                date_field_df2='discharge_date', 
                                                merge_on='id_patient')
        merge_adh_dysp
        merge_adh_dysp.rename(columns={'days_since_discharge_date':'days_since_last_dyspnea_test'}, inplace=True)

        period = 30
        aclq_timely = merge_adh_dysp[merge_adh_dysp.days_since_last_dyspnea_test <= period]
        aclq_late = merge_adh_dysp[merge_adh_dysp.days_since_last_dyspnea_test > period]
        aclq_timely
        aclq_timely_summary = pd.DataFrame()
        for (patient, survey_date), df in aclq_timely.groupby(['id_patient', 'survey_date']):
            aclq_timely_summary = aclq_timely_summary.append({
                        'id_patient': patient,
                        'survey_date': survey_date,
                        'medical_test': df.medical_test.sum(),
                        'days_since_last_dyspnea_test': df.iloc[-1]['days_since_last_dyspnea_test']
                    }, ignore_index=True)

        aclq_timely_summary
        # Merging adherence with aclq_timely_summary 
        merge_adh_dysp = adherence_change_analysis.merge(aclq_timely_summary, how='left', on=['id_patient', 'survey_date'])
        merge_adh_dysp.info()

        merge_adh_dysp['qualitative_result_change'] = merge_adh_dysp['qualitative_result_change'].fillna(0)
        merge_adh_dysp['days_since_last_control'] = merge_adh_dysp['days_since_last_control'].fillna(0)
        merge_adh_dysp['days_since_last_dyspnea_test'] = merge_adh_dysp['days_since_last_dyspnea_test'].fillna(0)
        merge_adh_dysp['medical_test'] = merge_adh_dysp['medical_test'].fillna(0)
        merge_all_dyspnea = merge_adh_dysp.copy()
        #Part 2: adherence + act datasets
        act_disag_cp.rename(columns={'result_date':'date_result'}, inplace = True)

        merge_adh_act = ovu.merge_on_closest_date(df1=adherence_change_analysis, 
                                          df2=act_disag_cp, 
                                          date_field_df1='survey_date', 
                                          date_field_df2='date_result', 
                                          merge_on='id_patient')

        merge_adh_act.rename(columns={'days_since_date_result':'days_since_act_test'}, inplace=True)
        period = 30
        aclq_timely_act = merge_adh_act[merge_adh_act.days_since_act_test <= period]
        aclq_late_act = merge_adh_act[merge_adh_act.days_since_act_test > period]

        aclq_timely_summary_act = pd.DataFrame()
        for (patient, survey_date), df in aclq_timely_act.groupby(['id_patient', 'survey_date']):
            aclq_timely_summary_act = aclq_timely_summary_act.append({
                'id_patient': patient,
                'survey_date': survey_date,
                'days_since_act_test': df.iloc[-1]['days_since_act_test'],
            # 'act_score':df.iloc[-1]['act_score'],
                #'result': df.iloc[-1]['result'],
                'A_ACT': df.A_ACT.sum(),
                'B_ACT': df.B_ACT.sum(),
                'C_ACT': df.C_ACT.sum(),
                'D_ACT': df.D_ACT.sum(),
                'FEEDBACK': df.iloc[-1]['FEEDBACK'],      
                
            }, ignore_index=True)
        merge_adh_ALL_act_dysp = merge_all_dyspnea.merge(aclq_timely_summary_act, how='left', on=['id_patient', 'survey_date'])    
        merge_adh_ALL_act_dysp['A_ACT'] = merge_adh_ALL_act_dysp['A_ACT'].fillna(0)
        merge_adh_ALL_act_dysp['B_ACT'] = merge_adh_ALL_act_dysp['B_ACT'].fillna(0)
        merge_adh_ALL_act_dysp['C_ACT'] = merge_adh_ALL_act_dysp['C_ACT'].fillna(0)
        merge_adh_ALL_act_dysp['D_ACT'] = merge_adh_ALL_act_dysp['D_ACT'].fillna(0)
        merge_adh_ALL_act_dysp['FEEDBACK'] = merge_adh_ALL_act_dysp['FEEDBACK'].fillna(0)
        merge_adh_ALL_act_dysp['days_since_act_test'] = merge_adh_ALL_act_dysp['days_since_act_test'].fillna(0)
        merge_modelable = merge_adh_ALL_act_dysp.copy()
        return merge_modelable

    def train(self):
        """Train two models: one Random forest classifier and one based on XGBoost"""
        if not self.rforest and not self.xgboost:
            covariates = self.data.columns[~self.data.columns.isin(['id_patient', 'qualitative_result', 'survey_date', 'num_reports','days_since_last_control',
                                                                            'qualitative_result_change','category','ongoing_adherence_percentage'])]
            print('> Training Random Forest classifier...')
            self.rforest = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
            self.rforest.fit(self.data[covariates], self.data['qualitative_result'])
            y_pred_rf = self.rforest.predict(self.data[covariates])
            accuracy_rf = accuracy_score(self.data['qualitative_result'], y_pred_rf)
            print("> Done. Accuracy: %.2f%%" % (accuracy_rf * 100.0))

            print('> Training XGBoost classifier...')
            params={
                'reg_alpha': 21, 
                'max_depth': 10, 
                'learning_rate': 0.05, 
                'gamma': 1.0
            }
            self.xgboost = XGBClassifier(random_state=0, booster='gbtree', **params)
            self.xgboost.fit(self.data[covariates], self.data['qualitative_result'])
            y_pred_xg = self.xgboost.predict(self.data[covariates])
            predictions_xg = [round(value) for value in y_pred_xg]
            accuracy_xg = accuracy_score(self.data['qualitative_result'], predictions_xg)
            print("> Done. Accuracy: %.2f%%" % (accuracy_xg * 100.0))
        return (self.rforest, self.xgboost)

    def dump_models(self):
        pickle.dump(self.rforest, open('clinical_forest.pkl','wb'))
        print('\n> Random forest classifier dumped into ./clinical_forest.pkl.')
        pickle.dump(self.xgboost, open('clinical_xgboost.pkl','wb'))
        print('> XGBoost classifier dumped into ./clinical_xgboost.pkl.')
