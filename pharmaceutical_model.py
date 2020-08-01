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

class pharmaceuticalModel():
    
    def __init__(self):
        self.data = self.load_data()
        self.rforest = None
        self.xgboost = None

    # Load pharmaceutical Datasets
    def load_data(self):
        adherence, adherence_change = ov.get_adherence_dataset()
        med_treat = ov.get_medical_treatments_dataset()
        bio_meds = ov.get_bio_meds_dataset()
        med_coll_issues = ov.get_med_collec_issues_dataset()

        # Adherencia Data Set
        adherence_change.id_patient =  adherence_change.id_patient.astype('str')
        fields = ['id_patient', 'survey_date', 'category', 
                 'qualitative_result', 'qualitative_result_change', 
                 'days_since_last_control', 'ongoing_adherence_percentage', 'num_reports']
        adherence_change_df = adherence_change[fields]

        # Biological medicines
        bio_meds['product_name'] = bio_meds['product_name'].astype(int)
        bio_meds.generic_name = bio_meds.generic_name.astype(int)

        # Medical collecting issues
        med_coll_issues.drop(columns='Observations', inplace=True)
        med_coll_issues.novelty_type = med_coll_issues.novelty_type.astype(int)
        med_coll_issues.event_type = med_coll_issues.event_type.astype('category')

        # Medical Treatments
        med_treat['quantity'] = med_treat['quantity'].fillna(0)
        med_treat.drop(columns=['diagnosis','medicine_name'], inplace=True)
        med_treat['regional_EPS'] = med_treat['regional_EPS'].astype('category')
        med_treat['medicine_code'] = med_treat['medicine_code'].astype('category')
        med_treat['diagnosis_code'] = med_treat['diagnosis_code'].astype('category')

        # Merging adherence and biological medicines 
        merge_adh_bio = ovu.merge_on_closest_date(df1=adherence_change_df, 
                                                df2=bio_meds, 
                                                date_field_df1='survey_date', 
                                                date_field_df2='record_date', 
                                                merge_on='id_patient')
        merge_adh_bio.rename(columns={'days_since_record_date':'days_since_last_delivery',
                             'quantity':'bio_quantity'}, inplace=True)
        period = 30
        aclq_timely = merge_adh_bio[merge_adh_bio.days_since_last_delivery <= period]
        aclq_late = merge_adh_bio[merge_adh_bio.days_since_last_delivery > period]

        aclq_timely_summary = pd.DataFrame()
        for (patient, survey_date), df in aclq_timely.groupby(['id_patient', 'survey_date']):
            aclq_timely_summary = aclq_timely_summary.append({
                'id_patient': patient,
                'survey_date': survey_date,
                'bio_quantity': df.bio_quantity.sum(),
                'days_since_last_delivery': df.iloc[-1]['days_since_last_delivery']
            }, ignore_index=True)

        # Merging adherence_bio Part 3
        merge_adh_bio = adherence_change_df.merge(aclq_timely_summary, how='left', on=['id_patient', 'survey_date'])

        merge_adh_bio['qualitative_result_change'] = merge_adh_bio['qualitative_result_change'].fillna(0)
        merge_adh_bio['days_since_last_control'] = merge_adh_bio['days_since_last_control'].fillna(0)
        merge_adh_bio['bio_quantity'] = merge_adh_bio['bio_quantity'].fillna(0)
        merge_adh_bio['days_since_last_delivery'] = merge_adh_bio['days_since_last_delivery'].fillna(0)

        merge_all_medicines = merge_adh_bio.copy()

        # Merging adherence_biological medicines and collecting issues 
        merge_adh_coll = ovu.merge_on_closest_date(df1=adherence_change_df, 
                                                df2=med_coll_issues, 
                                                date_field_df1='survey_date', 
                                                date_field_df2='register_date', 
                                                merge_on='id_patient')

        merge_adh_coll.rename(columns={'days_since_register_date':'days_since_last_coll_issue'}, inplace=True)

        period = 30
        aclq_timely_c = merge_adh_coll[merge_adh_coll.days_since_last_coll_issue <= period]
        aclq_late_c = merge_adh_coll[merge_adh_coll.days_since_last_coll_issue > period]
        aclq_timely_c

        aclq_timely_summary_c = pd.DataFrame()
        for (patient, survey_date), df in aclq_timely_c.groupby(['id_patient', 'survey_date']):
            aclq_timely_summary_c = aclq_timely_summary_c.append({
                'id_patient': patient,
                'survey_date': survey_date,
                'novelty_type': df.novelty_type.sum(),
                'days_since_last_coll_issue': df.iloc[-1]['days_since_last_coll_issue']
            }, ignore_index=True)
            
        merge_adh_coll = merge_all_medicines.merge(aclq_timely_summary_c, how='left', on=['id_patient', 'survey_date'])
        merge_adh_coll['days_since_last_coll_issue'] = merge_adh_coll['days_since_last_coll_issue'].fillna(0)
        merge_adh_coll['novelty_type'] = merge_adh_coll['novelty_type'].fillna(0)

        # Merging adherence_biological medicines_collecting issues and medical treatments
        merge_adh__med = ovu.merge_on_closest_date(df1=adherence_change_df, 
                                                df2=med_treat, 
                                                date_field_df1='survey_date', 
                                                date_field_df2='released_date', 
                                                merge_on='id_patient')
        biologicals_codes = ['M029140', 'M029751', 'M029551', 'M029157', 'M029631', 'M021755']
        merge_adh__med1 = merge_adh__med[~merge_adh__med['medicine_code'].isin(biologicals_codes)]
        merge_adh__med1

        merge_adh__med1.rename(columns={'days_since_released_date':'days_since_last_non_biological_delivery'}, inplace=True)

        period = 30
        aclq_timely_m = merge_adh__med1[merge_adh__med1.days_since_last_non_biological_delivery <= period]
        aclq_late_m = merge_adh__med1[merge_adh__med1.days_since_last_non_biological_delivery > period]
        aclq_timely_m

        aclq_timely_summary_m = pd.DataFrame()
        for (patient, survey_date), df in aclq_timely_m.groupby(['id_patient', 'survey_date']):
            aclq_timely_summary_m = aclq_timely_summary_m.append({
                'id_patient': patient,
                'survey_date': survey_date,
                'quantity_non_biological': df.quantity.sum(),
                'days_since_last_non_biological_delivery': df.iloc[-1]['days_since_last_non_biological_delivery']
            }, ignore_index=True)
            
        merge_all2 = merge_adh_coll.merge(aclq_timely_summary_m, how='left', on=['id_patient', 'survey_date'])
        merge_all2['days_since_last_non_biological_delivery'] = merge_all2['days_since_last_non_biological_delivery'].fillna(0)
        merge_all2['quantity_non_biological'] = merge_all2['quantity_non_biological'].fillna(0)
        merge_modelable = merge_all2.copy()
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
        pickle.dump(self.rforest, open('pharmaceutical_rforest.pkl','wb'))
        print('\n> Random forest classifier dumped into ./pharmaceutical_rfores.pkl.')
        pickle.dump(self.xgboost, open('pharmaceutical_xgboost.pkl','wb'))
        print('> XGBoost classifier dumped into ./pharmaceutical_xgboost.pkl.')






        




