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

class ExternalSourcesModel():
    
    def __init__(self):
        self.data = self.load_data()
        self.rforest = None
        self.xgboost = None
    
    
    def load_data(self): 
        # Load Datasets with extrenal sources data:
        pathologics = ov.get_pathological_record_dataset()
        adherence, adherence_change=ov.get_adherence_dataset()
        hospitalizations = ov.get_hospitalizations_dataset()
        emergencies=ov.get_emergencies_dataset()
        
        #Set the correct data types:
        #Adherence_change:
        adherence_change['category'] = adherence_change['category'].astype('category')
        adherence_change['category'].cat.reorder_categories(['N', 'N+', 'M', 'A-', 'A'], ordered=True, inplace=True)
        adherence_change['qualitative_result'] = adherence_change['qualitative_result'].astype(int)
        adherence_change['qualitative_result_change'] = adherence_change['qualitative_result_change'].fillna(0)
        adherence_change['qualitative_result_change'] = adherence_change['qualitative_result_change'].astype(int)
        adherence_change['ongoing_adherence_percentage'] = adherence_change['ongoing_adherence_percentage'].astype(int)
        select_fields = ['id_patient', 'survey_date', 'category', 'qualitative_result', 'qualitative_result_change', 'days_since_last_control', 'ongoing_adherence_percentage', 'num_reports']
        adherence_change_analysis = adherence_change[select_fields]

        #Emergencies:
        emergencies.drop(columns={'Unnamed: 0','diagnosis'}, inplace=True, axis=1)
        emergencies.description_service.unique()
        emergencies.loc[emergencies['description_service']=='ATENCION MEDICA PRIORITARIA', 'description_service'] = 1
        emergencies.loc[emergencies['description_service']=='CONSULTA MEDICO GENERAL NO PROGRAMADA', 'description_service'] = 0
        emergencies['description_service'] = emergencies['description_service'].astype(int)

        emergencies['diagnosis_code'] = emergencies['diagnosis_code'].astype('category')
        emergencies['registration_date']=pd.to_datetime(emergencies['registration_date'])

        #Pathologics:
        pathologics.rename(columns={'diagnosis_code': 'pathologics_diagnosis_code'}, inplace=True)
        pathologics['pathologics_diagnosis_code'] = pathologics['pathologics_diagnosis_code'].astype('category')

        #Hospitalizations:
        hospitalizations.rename(columns={'diagnosis_code': 'hospitalizations_diagnosis_code'}, inplace=True)
        hospitalizations.drop(columns=['gender', 'age'], axis=1, inplace=True)
        hospitalizations['in_date']=pd.to_datetime(hospitalizations['in_date'])
        hospitalizations['out_date']=pd.to_datetime(hospitalizations['out_date'])

        pathologics.drop(columns=['end_date'], axis=1, inplace=True)
        emergencies.drop(columns=['code_service'], axis=1, inplace=True)



        #Merge:
        merged_adherence = adherence_change_analysis.merge(emergencies, how='left', on='id_patient')
        merged_adherence = merged_adherence.merge(pathologics, how='left', on='id_patient')
        merged_adherence = merged_adherence.merge(hospitalizations, how='left', on='id_patient')
        
        #Reorder the dataset:
        cols= list(merged_adherence.columns.values);
        new_oder= [0, 1, 9, 13, 14, 20, 2,3,4,5,6,7,8,10,11,12,15,16,17,18,19,21];
        merged_adherence_1=merged_adherence[merged_adherence.columns[new_oder]];

        #Sort the dataset by dates:
        merged_adherence_1.sort_values(by=['id_patient',
                                       'survey_date',
                                       'registration_date',
                                       'update_date',
                                       'start_date','in_date'],ascending=True);
        # merged_adherence_1.info();
        cols1= list(merged_adherence_1.columns.values);
        merged_adherence_1[['id_patient','qualitative_result','qualitative_result_change','days_since_last_control','ongoing_adherence_percentage','num_reports','description_service','icu_days','scu_days','days']]=merged_adherence_1[['id_patient','qualitative_result','qualitative_result_change','days_since_last_control','ongoing_adherence_percentage','num_reports','description_service','icu_days','scu_days','days']].fillna(0)

        #Process hospitalizations data:
        #Part 1.
        adherence_change_hospitalizations = ovu.merge_on_closest_date(df1=adherence_change_analysis, df2=hospitalizations, date_field_df1='survey_date', date_field_df2='in_date', merge_on='id_patient')
        adherence_change_hospitalizations.rename(columns={'days_since_in_date': 'days_since_last_hospitalization'}, inplace=True)
        period = 30
        aclq_timely = adherence_change_hospitalizations[adherence_change_hospitalizations.days_since_last_hospitalization <= period]
        aclq_late = adherence_change_hospitalizations[adherence_change_hospitalizations.days_since_last_hospitalization > period]
        aclq_timely
        #Part 2.
        aclq_timely_summary = pd.DataFrame()
        for (patient, survey_date), df in aclq_timely.groupby(['id_patient', 'survey_date']):
            aclq_timely_summary = aclq_timely_summary.append({
                'id_patient': patient,
                'survey_date': survey_date,
                'num_hospitalizations_last_30_days': df.shape[0],
                'total_days': df.days.sum(),
                'days_since_last_hospitalization': df.iloc[-1]['days_since_last_hospitalization']
            }, ignore_index=True)
        #Part 3.
        hosp_patho_emer_adherence = merged_adherence_1.merge(aclq_timely_summary, how='left', on=['id_patient', 'survey_date'])
        #Part 4.
        hosp_patho_emer_adherence['total_days'] = hosp_patho_emer_adherence['total_days'].fillna(0)
        hosp_patho_emer_adherence['num_hospitalizations_last_30_days'] = hosp_patho_emer_adherence['num_hospitalizations_last_30_days'].fillna(0)
        hosp_patho_emer_adherence['days_since_last_hospitalization'] = hosp_patho_emer_adherence['days_since_last_hospitalization'].fillna(0)

        #Process emergencies data:
        # Part 1.
        adherence_change_emergencies= ovu.merge_on_closest_date(df1=adherence_change_analysis, df2=emergencies, date_field_df1='survey_date', date_field_df2='registration_date', merge_on='id_patient')
        adherence_change_emergencies.rename(columns={'days_since_registration_date': 'days_since_last_emergency'}, inplace=True)
        period = 30
        aclq_timely_e = adherence_change_emergencies[adherence_change_emergencies.days_since_last_emergency <= period]
        aclq_late_e = adherence_change_emergencies[adherence_change_emergencies.days_since_last_emergency > period]
        aclq_timely_e

        # Part 2.
        aclq_timely_e_summary = pd.DataFrame()
        for (patient, survey_date), df in aclq_timely_e.groupby(['id_patient', 'survey_date']):
            aclq_timely_e_summary = aclq_timely_e_summary.append({
                'id_patient': patient,
                'survey_date': survey_date,
                'num_emergencies_last_30_days': df.shape[0],
                'days_since_last_emergency': df.iloc[-1]['days_since_last_emergency']
            }, ignore_index=True)

        # Part 3.
        hosp_patho_emer_adherence_2 = hosp_patho_emer_adherence.merge(aclq_timely_e_summary, how='left', on=['id_patient', 'survey_date'])

        # Part 4.
        hosp_patho_emer_adherence_2['days_since_last_emergency'] = hosp_patho_emer_adherence_2['days_since_last_emergency'].fillna(0)
        hosp_patho_emer_adherence_2['num_emergencies_last_30_days'] = hosp_patho_emer_adherence_2['num_emergencies_last_30_days'].fillna(0)
        hosp_patho_emer_adherence_2['days_since_last_hospitalization'] = hosp_patho_emer_adherence_2['days_since_last_hospitalization'].fillna(0)

        #Drop 'category' column:

        hosp_patho_emer_adherence_2.drop(columns=['category', 'qualitative_result_change','health_provider','regional_health_provider'], axis=1, inplace=True)

        #from all the 'pathologics_diagnosis_code' use the 7 most common as 1 and the rest as 0,
        #being 1 more important than 0.
        hosp_patho_emer_adherence_2['pathologics_diagnosis_code']= np.where(
        (hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] == 'J459') |
        (hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] == 'J46X') |
        (hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] == 'J189') |
        (hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] == 'J441') |
        (hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] == 'J159') |
        (hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] == 'N390') |
        (hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] == 'J450'), 1,
        hosp_patho_emer_adherence_2['pathologics_diagnosis_code'])
        hosp_patho_emer_adherence_2['pathologics_diagnosis_code']= np.where(
        (hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] != 1), 0,
        hosp_patho_emer_adherence_2['pathologics_diagnosis_code'])

        #from all the 'hospitalizations_diagnosis_code' use the 7 most common as 1 and the rest as 0,
        #being 1 more important than 0.
        hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code']= np.where(
        (hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] == 'J450') |
        (hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] == 'J459') |
        (hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] == 'JL509') |
        (hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] == 'I10X') |
        (hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] == 'J304') |
        (hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] == 'L501') |
        (hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] == 'L500'), 1,
        hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'])
        hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code']= np.where(
        (hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] != 1), 0,
        hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'])

        hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'] = hosp_patho_emer_adherence_2['hospitalizations_diagnosis_code'].astype(int)
        hosp_patho_emer_adherence_2['pathologics_diagnosis_code'] = hosp_patho_emer_adherence_2['pathologics_diagnosis_code'].astype(int)



        # Get a copy of the previous dataset to encode the features

        ho_pa_em_ad_modelable = hosp_patho_emer_adherence_2.copy()
        ho_pa_em_ad_modelable['diagnosis_code'] = ho_pa_em_ad_modelable['diagnosis_code'].cat.codes

        return ho_pa_em_ad_modelable
    
    def train(self):
        """Train two models: one Random forest classifier and one based on XGBoost"""
        if not self.rforest and not self.xgboost:
            covariates = self.data.columns[~self.data.columns.isin(['id_patient', 
                                                                                 'qualitative_result',
                                                                                 'registration_date',
                                                                                 'update_date',
                                                                                 'start_date',
                                                                                 'in_date',
                                                                                 'out_date',
                                                                                 'survey_date'])]
            
            print('> Training Random Forest classifier... update')
            self.rforest = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
            self.rforest.fit(self.data[covariates], self.data['qualitative_result'])
            y_pred_rf = self.rforest.predict(self.data[covariates])
            accuracy_rf = accuracy_score(self.data['qualitative_result'], y_pred_rf)
            print("> Done. Accuracy: %.2f%%" % (accuracy_rf * 100.0))

            print('> Training XGBoost classifier...')
            params={
                'reg_alpha': 20, 
                'max_depth': 12, 
                'learning_rate': 0.05, 
                'gamma': 0.2
            }
            self.xgboost = XGBClassifier(random_state=0, booster='gbtree', **params)
            self.xgboost.fit(self.data[covariates], self.data['qualitative_result'])
            y_pred_xg = self.xgboost.predict(self.data[covariates])
            predictions_xg = [round(value) for value in y_pred_xg]
            accuracy_xg = accuracy_score(self.data['qualitative_result'], predictions_xg)
            print("> Done. Accuracy: %.2f%%" % (accuracy_xg * 100.0))
        
        return (self.rforest, self.xgboost)

    def dump_models(self):
        pickle.dump(self.rforest, open('external_sources_rforest.pkl','wb'))
        print('\n> Random forest classifier dumped into ./external_sources_rfores.pkl.')
        pickle.dump(self.xgboost, open('external_sources_xgboost.pkl','wb'))
        print('> XGBoost classifier dumped into ./external_sources_xgboost.pkl.')
    
