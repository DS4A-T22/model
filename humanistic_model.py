import sys
import os
sys.path.append("../")
import numpy as np
import pandas as pd
import pickle
import requests
import json
from exploratory_data_analysis import omnivida_loader as ov
from exploratory_data_analysis import omnivida_util as ovu
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier

class HumanisticModel():

    def __init__(self):
        self.data = self.load_data()
        self.rforest = None
        self.xgboost = None

    # Load Datasets with humanistic data

    def load_data(self):
        _, adherence_change = ov.get_adherence_dataset()
        basic_info = ov.get_basic_info_dataset()
        familiar_records = ov.get_family_records_dataset()
        life_quality = ov.get_wellbeing_index_dataset()

        # Filter adherence dataframe to include only relevant fields

        select_fields = ['id_patient', 'survey_date', 'category', 'qualitative_result', 'qualitative_result_change', 'days_since_last_control', 'ongoing_adherence_percentage', 'num_reports']
        adherence_change_analysis = adherence_change[select_fields]

        # Merge Adherence and Basic info data

        bi_adherence = adherence_change_analysis.merge(basic_info, how='left', on='id_patient')
        bi_adherence['category'] = bi_adherence['category'].astype('category')
        bi_adherence['category'].cat.reorder_categories(['N', 'N+', 'M', 'A-', 'A'], ordered=True, inplace=True)
        bi_adherence['age_at_survey_date'] = (np.ceil((bi_adherence['survey_date'] - bi_adherence['birthdate']) / np.timedelta64(1, 'Y'))).astype(int)
        bi_adherence.drop(columns=['city', 'department', 'age'], axis=1, inplace=True)

        # Process family record data

        respiratory_related_diagnosis = ['ASMA MIXTA', 'ASMA NO ALERGICA', 'ASMA, NO ESPECIFICADA', 
                                        'CARCINOMA IN SITU DEL BRONQUIO Y DEL PULMON', 'ENFERMEDAD PULMONAR OBSTRUCTIVA CRONICA, NO ESPECIFICADA']

        consolidated_family_record = pd.DataFrame()
        for paciente, df in familiar_records.groupby('id_patient'):
            temp_df = df.copy()
            row = {}
            row['id_patient'] = int(paciente)
            row['family_history_reported'] = 1
            row['num_family_records'] = temp_df.shape[0]
            row['family_respiratory_related_diagnosis'] = len(temp_df[temp_df.diagnosis.isin(respiratory_related_diagnosis)])
            row['family_non_respiratory_related_diagnosis'] = len(temp_df[~temp_df.diagnosis.isin(respiratory_related_diagnosis)])
            consolidated_family_record = consolidated_family_record.append(row, ignore_index=True)

        ## Rearrange columns
        consolidated_family_record = consolidated_family_record[
            ['id_patient', 'family_respiratory_related_diagnosis', 'family_non_respiratory_related_diagnosis']
        ]

        # Merge Adherence, Basic info data and Family records data

        bi_fr_adherence = bi_adherence.merge(consolidated_family_record, how='left', on='id_patient')
        bi_fr_adherence['family_respiratory_related_diagnosis'] = bi_fr_adherence['family_respiratory_related_diagnosis'].fillna(0)
        bi_fr_adherence['family_non_respiratory_related_diagnosis'] = bi_fr_adherence['family_non_respiratory_related_diagnosis'].fillna(0)

        # Process wel-being index data

        period = 30

        pivoted_life_quality = pd.DataFrame()
        for patient, df in life_quality.groupby('id_patient'):
            pvt = df.pivot(index='creation_date', columns='dimension', values='score')
            pvt.columns = pvt.columns.categories
            pvt.reset_index(inplace=True)
            pvt['id_patient'] = patient
            cols = [list(pvt.columns)[-1]] + list(pvt.columns)[:-1]
            pvt = pvt[cols]
            pivoted_life_quality = pivoted_life_quality.append(pvt, ignore_index=True)

        pivoted_life_quality.columns = ['id_patient', 'creation_date', 'personal_environment', 'psychological_health', 'interpersonal_relationships', 'physical_health']
        pivoted_life_quality['wb_score'] = (pivoted_life_quality['personal_environment'] + \
                pivoted_life_quality['psychological_health'] + \
                pivoted_life_quality['interpersonal_relationships'] + \
                pivoted_life_quality['physical_health']) / 4.

        adherence_change_life_quality = ovu.merge_on_closest_date(df1=adherence_change_analysis, df2=pivoted_life_quality, date_field_df1='survey_date', date_field_df2='creation_date', merge_on='id_patient')
        adherence_change_life_quality.rename(columns={'days_since_creation_date': 'days_since_last_well_being_survey'}, inplace=True)

        aclq_timely = adherence_change_life_quality[adherence_change_life_quality.days_since_last_well_being_survey <= period]
        aclq_late = adherence_change_life_quality[adherence_change_life_quality.days_since_last_well_being_survey > period]

        aclq_timely_summary = pd.DataFrame()
        for (patient, survey_date), df in aclq_timely.groupby(['id_patient', 'survey_date']):
            aclq_timely_summary = aclq_timely_summary.append({
                'id_patient': patient,
                'survey_date': survey_date,
                'num_wb_reports_last_30_days': df.shape[0],
                'wb_score': df.iloc[-1]['wb_score'],
                'days_since_last_wb_survey': df.iloc[-1]['days_since_last_well_being_survey']
            }, ignore_index=True)

        # Merge Adherence, Basic info data, Family records and Well-being data

        bi_fr_wb_adherence = bi_fr_adherence.merge(aclq_timely_summary, how='left', on=['id_patient', 'survey_date'])
        bi_fr_wb_adherence['days_since_last_wb_survey'] = bi_fr_wb_adherence['days_since_last_wb_survey'].fillna(0)
        bi_fr_wb_adherence['days_since_last_control'] = bi_fr_wb_adherence['days_since_last_control'].fillna(0)
        bi_fr_wb_adherence['num_wb_reports_last_30_days'] = bi_fr_wb_adherence['num_wb_reports_last_30_days'].fillna(0)
        bi_fr_wb_adherence['wb_score'] = bi_fr_wb_adherence['wb_score'].fillna(0)
        bi_fr_wb_adherence.drop(columns=['category', 'qualitative_result_change', 'social_security_regime', 'zone', 'social_security_affiliation_type', 'employment_type'], axis=1, inplace=True)

        # Get a copy of the previous dataset to encode the features

        bi_fr_wb_adherence_modelable = bi_fr_wb_adherence.copy()
        # bi_fr_wb_adherence_modelable['category'] = bi_fr_wb_adherence_modelable['category'].cat.codes
        # bi_fr_wb_adherence_modelable['gender'] = bi_fr_wb_adherence_modelable['gender'].cat.codes
        bi_fr_wb_adherence_modelable['education'] = bi_fr_wb_adherence_modelable['education'].cat.codes
        # bi_fr_wb_adherence_modelable['civil_status'] = bi_fr_wb_adherence_modelable['civil_status'].cat.codes
        bi_fr_wb_adherence_modelable['socioeconomic_level'] = bi_fr_wb_adherence_modelable['socioeconomic_level'].cat.codes
        # bi_fr_wb_adherence_modelable['occupation'] = bi_fr_wb_adherence_modelable['occupation'].cat.codes
        bi_fr_wb_adherence_modelable = pd.get_dummies(bi_fr_wb_adherence_modelable, columns=['gender', 'civil_status', 'occupation'])
        # Setting Non-adherent label as '1'
        bi_fr_wb_adherence_modelable['qualitative_result'] = np.logical_xor(bi_fr_wb_adherence_modelable['qualitative_result'],1).astype(int)
        return bi_fr_wb_adherence_modelable

    def train(self):
        """Train two models: one Random forest classifier and one based on XGBoost"""
        if not self.rforest and not self.xgboost:
            covariates = self.data.columns[~self.data.columns.isin(['id_patient', 'qualitative_result', 'survey_date', 'birthdate', 'num_reports'])]
            print('> Training Random Forest classifier...')
            self.rforest = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
            self.rforest.fit(self.data[covariates], self.data['qualitative_result'])
            y_pred_rf = self.rforest.predict(self.data[covariates])
            accuracy_rf = accuracy_score(self.data['qualitative_result'], y_pred_rf)
            print("> Done. Accuracy: %.2f%%" % (accuracy_rf * 100.0))

            print('> Training XGBoost classifier...')
            params={
                'reg_alpha': 23, 
                'max_depth': 8, 
                'learning_rate': 0.01, 
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
        pickle.dump(self.rforest, open('human_rforest.pkl','wb'))
        print('\n> Random forest classifier dumped into ./human_rfores.pkl.')
        pickle.dump(self.xgboost, open('human_xgboost.pkl','wb'))
        print('> XGBoost classifier dumped into ./human_xgboost.pkl.')
        

