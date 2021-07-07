import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
#import pandas_profiling
#from scipy import stats
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression, LinearRegression
#from sklearn.model_selection import GridSearchCV
#from sklearn import metrics
low_memory = False

'''
The purpose of this file is to work with the dependent variables of the bdd data set to organize them for 
analysis.

'''
def csv_to_dict():  #First step: formatting column headers consistently
    '''Reads in multiple .csv files and returns a dictionary of dataframes.'''
    df_dict = {}
    df_dict['blood_dp'] = pd.read_csv('data/raw/blood_data_points_6-27-19.csv', index_col=False)
    #Merges all the blood_data points csvs into one large dataframe
    df_dict['blood_icd9'] = pd.read_csv('data/raw/blood_icd_dependent_6-27-19.csv', index_col=False)
    df_dict['blood_dp_vent'] = pd.read_csv('data/raw/blood_pt_with_vent_days_6-27-19.csv', index_col=False)

    return df_dict

def format_dict(df_dict):
    '''Precond: Takes in a dictionary of dataframes and formats the column headers to a uniform format.
    Returns a dictionary with formatted dataframes.'''
    blood_dp = df_dict['blood_dp']
    blood_dp.columns = blood_dp.columns.str.strip().str.lower().str.replace(' ', '_')
    df_dict['blood_dp'] = blood_dp
    blood_dp.columns = blood_dp.columns.str.strip().str.lower().str.replace(' ', '_')
    dp_vent = df_dict['blood_dp_vent']
    dp_vent.columns = dp_vent.columns.str.strip().str.lower().str.replace(' ', '_')
    df_dict['blood_dp_vent'] = dp_vent
    icd9 = df_dict['blood_icd9']
    icd9.columns = icd9.columns.str.strip().str.lower().str.replace(' ', '_')
    df_dict['blood_icd9'] = icd9
    return df_dict

def consolidate_vent_days(df_dict):
    '''Precond: Takes in a dictionary of dataframes.
    This function formats the ventilation days entry of the dataframe dict and consolidates
    the patients with their total ventilation days in order to eliminate repeat entries.
    Returns the same formatted dataframe dictionary.'''
    dp_vent = df_dict['blood_dp_vent']
    vent = dp_vent[['person_id', 'actualventdays']].copy() #Dataframe that holds the vent days for a person.
    vent = vent.groupby('person_id').sum()
    df_dict['blood_dp_vent'] = vent
    df_dict['blood_dp'] = pd.merge(df_dict['blood_dp'], vent, on='person_id', how='outer') #Merges vent days with patients that have relevant vent days, merged in outer style to keep patients who don't have vent days.
    return df_dict

def icd9_check(df_dict):
    '''Precond: Takes in a dictionary of dataframes with the main blood_dp
    dataframe formatted to include total ventilation days.
    
    This function creates boolean values saved as zeroes or ones for every patient entry in the icd9_dependent dataframe
    and saves it to a new formatted dataframe called diagnoses_results with the patient ID as the row label, and diagnoses as zeroes or ones.

    It then merges the dataframe with the blood_dp dataframe for the patients with diagnoses, patients that do not have diagnoses are set values of zero.
    '''
    icd9 = df_dict['blood_icd9']
    icd9 = icd9[icd9['icd_dependent'] == 'Yes'] #Only patients with icd9 values will show.
    code_dict = {'myocard':['410', '41001', '41002', '4101'], 'UTI': ['V1302', '99664'],\
                 'surg_inf' : ['99664'], 'sepsis' : ['99591', '99592', '78552'], 'vent_pneu' : ['99731'],\
                 'kid_inj' : ['5849'],'resp_dist' : ['51882'],'organ_fail' : ['51882'],'vein_throm': ['4534'],\
                 'pulm_emb' : ['4151','51882'], 'altered_mental' : ['293', '3483'], 'transf_react' : ['5187'],'cen_line_inf' : ['99931']}
    #This dictionary of icd9 codes is utilized for checking patients that have these codes.
    diags = {}
    icd9 = icd9.drop(['icd_dependent'], axis=1)
    for index, row in icd9.iterrows():
        diags[row['person_id']] = [row['addg{}'.format(i + 1)] for i in range(len(row)) if i not in [6,7,8]] ##6/27/2019 data-- Version removed columns 7,8, & 9.
        #diags[row['person_id']] = row[['addg{}'.format(i + 1 for i in range(25))]]] #List of addgs to loop through for every patient.
    patients = {} #This will be a dictionary of dictionaries, with the patient_ID being the key
    for key in diags: #For each patient
        diagnoses = {'myocard' : 0, 'UTI' : 0, 'surg_inf' : 0 ,'sepsis' : 0,'vent_pneu' : 0 ,'kid_inj' : 0, 'resp_dist' : 0, 'organ_fail' : 0,\
                    'vein_throm' : 0, 'pulm_emb' : 0, 'altered_mental' : 0, 'transf_react' : 0, 'cen_line_inf' : 0}
        for diag in code_dict: #For each diagnoses name
            for item in code_dict[diag]: #For each code in the diagnosis list
                if item in diags[key]:
                    diagnoses[diag] = 1 #If the patient has the code in their list of ADDGs, it will set their diagnoses value to 1.
        patients[key] = diagnoses
    diagnoses_results = pd.DataFrame.from_dict(patients, orient='index')
    diagnoses_results['person_id'] = diagnoses_results.index
    df_dict['blood_dp'] = pd.merge(df_dict['blood_dp'], diagnoses_results, on='person_id') ##Adds diagnoses to blood_dp patient column
    return df_dict

def sort_by_genders(df_dict):
    '''Precond: Takes in a dictionary of dataframes.
    Organizes the blood_dp dataframe by specified analysis values required.
    Then further organizes the data by separating them into specified sex:sex dyads.
    It returns a dictionary of sorted sex:sex dyad dataframes.'''
    blood_dp = df_dict['blood_dp']
    blood_dp['bddiff'] = (blood_dp.donor_age - blood_dp.age)
    blood_dp['mortality'] = (blood_dp.expired == 0).astype('str')
    blood_dp.loc[blood_dp['expired'] == 0, 'mortality'] = "Alive"
    blood_dp.loc[blood_dp['expired'] == 1, 'mortality'] = "Dead"
    blood_dp = blood_dp[blood_dp.age >= 18]
    blood_dp['blood_type'] = ""
    blood_dp.loc[blood_dp['description'].str.contains("Platelet"), 'blood_type'] = "Platelets"
    blood_dp.loc[blood_dp['description'].str.contains("Red Blood Cells"), 'blood_type'] = "Packed Red Blood Cells"
    blood_dp.loc[blood_dp['description'].str.contains("Plasma"), 'blood_type'] = "Fresh Frozen Plasma"
    blood_dp_dep = blood_dp[['person_id', 'mortality', 'bddiff', 'icu_days', 'los_days', 'actualventdays', 'myocard', 'UTI', 'surg_inf',\
         'sepsis', 'vent_pneu', 'kid_inj', 'resp_dist', 'vein_throm', 'organ_fail', 'pulm_emb',\
         'altered_mental', 'transf_react', 'cen_line_inf', 'gender', 'donor_sex', 'blood_type']].copy()
    #This collects all of the dependent variables listed in the word doc.
    averaged = blood_dp_dep.groupby(['person_id','mortality','blood_type']).mean()
    m_m = blood_dp_dep[(blood_dp_dep['gender']=='Male') & (blood_dp_dep['donor_sex']=='Male')]
    m_f = blood_dp_dep[(blood_dp_dep['gender']=='Male') & (blood_dp_dep['donor_sex']=='Female')]
    f_f = blood_dp_dep[(blood_dp_dep['gender']=='Female') & (blood_dp_dep['donor_sex']=='Female')]
    f_m = blood_dp_dep[(blood_dp_dep['gender']=='Female') & (blood_dp_dep['donor_sex']=='Male')]
    sex_sorted = {'Male accepted Male': m_m, 'Male accepted Female': m_f, 'Female accepted Female': f_f, 'Female accepted Male': f_m}
    return (sex_sorted, averaged)

def get_nonsorted_stats(average):
    '''Takes in a dataframe of averaged blood data points ignoring sex dyads.
        The purpose of this function is to compare all of the datapoints located in the dataset and do create a visualization to see where each
        data point lies before setting them into sex:sex dyads.'''
    blood_pckgs = ["Platelets", "Packed Red Blood Cells", "Fresh Frozen Plasma"]
    results = open("data/graphs/Secondary Outcomes/UTI/UTI Results.txt", "a+")
    for package in blood_pckgs:    
        averaged = average.copy()
        averaged = averaged.reset_index()
        averaged = averaged[averaged['blood_type'].str.contains(package)]
        averaged = (averaged.groupby(['bddiff', 'UTI'])['person_id'].count())
        averaged = averaged.reset_index()
        target = averaged['UTI']
        N = averaged['person_id'].sum()
        sns.set(font_scale=2)
        fg = sns.lmplot(x='bddiff', data=averaged, y='UTI', y_jitter=.01, x_jitter=.15, logistic=True, scatter_kws={'alpha':0.05}, line_kws={'color':'red'})
        fg.fig.set_size_inches(15,10)
        fg.set(ylabel = "Patients with a UTI", xlabel = "Blood Donor Age Differential (Donor - Recipient)", title= "Unsorted Overall Logistic Regression Model of UTI Rate vs Age differential where Patients Receive "+ package)
        plt.show()
        fg.savefig('data/graphs/Secondary Outcomes/UTI/Overall UTI Log Reg with '+ package)
        log_reg = sm.Logit(target,sm.add_constant(averaged.bddiff))
        result = log_reg.fit()
        results.write("Overall with " + package + '\n')
        results.write("Total amount of unique patients (N) =  "+ str(N) +'\n')
        results.write(str(result.summary())+ '\n')
        '''THIS BOTTOM CODE IS USED FOR LINEAR REGRESSION
        -----------------------------------------------------------------------------
        pearson_coef, p_value = stats.pearsonr(averaged.bddiff, averaged.icu_days)
        X = averaged['bddiff'].values.reshape(-1,1)
        Y = averaged['icu_days'].values.reshape(-1,1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        regressor = LinearRegression()  
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        predictions = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
        r_squared = metrics.r2_score(y_pred = predictions['Predicted'], y_true = predictions['Actual'])
        plt.scatter(X_test, y_test,  color='gray')
        plt.plot(X_test, y_pred, color='red', linewidth=2)
        plt.suptitle("Unsorted Linear Regression Model of ICU Days vs Age differential where Patients Receive "+ package)
        plt.xlabel("Blood Donor Age Differential (Donor - Recipient)")
        plt.ylabel("ICU Days")
        #plt.set(ylabel = "Lenth of Stay (in days)", xlabel = "Blood Donor Age Differential (Donor - Recipient)", title= "Unsorted Linear Regression Model of Length of Stay vs Age differential where Patients Receive "+ package)
        plt.savefig('data/graphs/Primary Outcomes/Number of ICU Days/Overall/Unsorted ICU days with '+ package)
        plt.show()
        results.write('-----------------------------------------------\n')
        results.write("Overall (unsorted data) with " + package + '\n')
        #To retrieve the intercept:
        results.write("The regressor intercept is: " + str(regressor.intercept_) + '\n')
        #For retrieving the slope:
        results.write('The Regressor coefficient is: ' + str(regressor.coef_) + '\n')
        results.write("The coefficient of determination (r^2) = " + str(r_squared) + '\n')
        results.write('The pearson coefficient is: ' + str(pearson_coef) + '\n')
        results.write('The p_value is: ' + str(p_value) + '\n')
        results.write('Mean Absolute Error:' + str(metrics.mean_absolute_error(y_test, y_pred)) +'\n')  
        results.write('Mean Squared Error:' + str(metrics.mean_squared_error(y_test, y_pred)) + '\n')  
        results.write('Root Mean Squared Error:' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) + '\n')
        results.write('-----------------------------------------------\n\n\n\n')
        '''
def get_stats(sex_sorted):
    '''Precond: Takes in a dictionary of sex:sex separated dataframes of blood datapoints.

    The purpose of this function is to create a dataframe that takes in every unique patient,
    and calculates the required stats for the primary outcomes,
    then writes the results to a separate .csv.

    By the study's specifications, for each sex:sex dyad dataframe that has multiple entries for the same patient,
    the resulting age differential is calculated by pooling the ages, then averaging them by their total for that patient.

    VARS TO WORK WITH:
        [['gender', 'donor_sex','person_id', 'expired', 'icu_days', 'los_days', 'actualventdays', 'myocard', 'UTI', 'surg_inf',\
         'sepsis', 'vent_pneu', 'kid_inj', 'resp_dist', 'vein_throm', 'organ_fail', 'pulm_emb',\
         'altered_mental', 'transf_react', 'cen_line_inf']]'''
    blood_pckgs = ["Platelets", "Packed Red Blood Cells", "Fresh Frozen Plasma"]
    bins = np.arange(-90, 95, 5)
    results = open("data/graphs/Secondary Outcomes/UTI/UTI Results.txt", "a+")
    for package in blood_pckgs:
        dyads = {}
        for sex_sorted_dyad in sex_sorted.keys():
            dyad = sex_sorted[sex_sorted_dyad]
            averaged = dyad[dyad['blood_type'].str.contains(package)]
            averaged = averaged.groupby(['person_id','UTI']).mean()
            averaged = averaged.reset_index()
            averaged = (averaged.groupby(['bddiff', 'UTI'])['person_id'].count())
            averaged = averaged.reset_index()
            dyads[sex_sorted_dyad] = averaged
        for dyad in dyads:
            N = dyads[dyad]['person_id'].sum()
            target = dyads[dyad]['UTI']
            fg = sns.lmplot(x='bddiff', data=dyads[dyad], y='UTI', y_jitter=.01, x_jitter=.15, logistic=True, scatter_kws={'alpha':0.05}, line_kws={'color':'red'})
            fg.fig.set_size_inches(15,5)
            fg.set(ylabel = "UTI", xlabel = "Blood Donor Age Differential (Donor - Recipient)", title= "Overall Logistic Regression Model of UTI Rate vs Age differential where {} and patients recieve {}".format(str(dyad), package))
            plt.show()
            fg.savefig('data/graphs/Secondary Outcomes/UTI/'+dyad+'/'+dyad+' UTI Log Reg '+ package)
            log_reg = sm.Logit(target,sm.add_constant(dyads[dyad].bddiff))
            result = log_reg.fit()
            results.write(dyad + ' ' + package +'\n')
            results.write("Total amount of unique patients (N) =  "+ str(N) + '\n')
            results.write(str(result.summary())+ '\n')
    results.close()
    '''THIS BOTTOM CODE IS USED FOR LINEAR REGRESSION
    ----------------------------------------------------------------------------
            pearson_coef, p_value = stats.pearsonr(dyads[dyad].bddiff, dyads[dyad].icu_days)
            X = dyads[dyad]['bddiff'].values.reshape(-1,1)
            Y = dyads[dyad]['icu_days'].values.reshape(-1,1)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
            regressor = LinearRegression()  
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            predictions = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
            r_squared = metrics.r2_score(y_pred = predictions['Predicted'], y_true = predictions['Actual'])
            plt.scatter(X_test, y_test,  color='gray')
            plt.plot(X_test, y_pred, color='red', linewidth=2)
            plt.suptitle("Linear Regression Model of Days in ICU vs Age differential where {} and patients recieve {}".format(str(dyad), package))
            plt.xlabel("Blood Donor Age Differential (Donor - Recipient)")
            plt.ylabel("ICU Days")
           # plt.set(ylabel = "Lenth of Stay (in days)", xlabel = "Blood Donor Age Differential (Donor - Recipient)", title= "Linear Regression Model of Length of Stay vs Age differential where {} and patients recieve {}".format(str(dyad), package))
            plt.savefig('data/graphs/Primary Outcomes/Number of ICU Days/'+dyad+'/'+dyad+' ICU Days with '+ package)
            plt.show()
            results.write('-----------------------------------------------\n')
            results.write(dyad + " data with " + package + '\n')
            #To retrieve the intercept:
            results.write("The regressor intercept is: " + str(regressor.intercept_) + '\n')
            #For retrieving the slope: 
            results.write('The Regressor coefficient is: ' + str(regressor.coef_) + '\n')
            results.write("The coefficient of determination (r^2) = " + str(r_squared) + '\n')
            results.write('The pearson coefficient is: ' + str(pearson_coef) + '\n')
            results.write('The p_value is: ' + str(p_value) + '\n')
            results.write('Mean Absolute Error:'+ str(metrics.mean_absolute_error(y_test, y_pred)) +'\n')  
            results.write('Mean Squared Error:'+ str(metrics.mean_squared_error(y_test, y_pred)) + '\n')  
            results.write('Root Mean Squared Error:' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) + '\n')
            results.write('-----------------------------------------------\n\n\n\n')
            '''
    results.close()
#def make_visuals(df_collection):

def main():
    df_dict = csv_to_dict()
    df_dict = format_dict(df_dict)
    df_dict = consolidate_vent_days(df_dict)
    df_dict = icd9_check(df_dict)
    sex_sorted = sort_by_genders(df_dict)
    get_nonsorted_stats(sex_sorted[1])
    get_stats(sex_sorted[0])

main()
