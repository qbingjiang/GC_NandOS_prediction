import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import pandas as pd

path = r'.'
training_shanxi_set1 = path + r'/pred_train.xlsx' 
df_features_shanxi_1 = pd.read_excel(training_shanxi_set1 ) 
df_features_shanxi_1['target_'] = df_features_shanxi_1['淋巴结转移（0为N0,1为N+）']
X_train = df_features_shanxi_1 
column_sel_e = ['Age', 'BMI', 'Sex', 
                # 'T stage', 'N stage',
                'Locations', 'Differentiation', 
                'Borrmann', 'Lauren', 
                'Pre-NACT CEA', 'Pre-NACT CA199', 
                'Regimens', 'Course' 
                ] 
X_train  = X_train[column_sel_e + ['target_'] ]
data = X_train

test_class = [
              'continuous',  
              'continuous',  
              'categorical', 
              'categorical', 
              'categorical', 
              'categorical', 
              'categorical', 
              'categorical', 
              'categorical', 
              'categorical', 
              'categorical', 
              'categorical', 
         ]


column_sel = column_sel_e 
for col in column_sel[2:]: 
    data[col] = data[col].astype('category')
# dummy_variables = pd.get_dummies(data[column_sel[2:]], prefix=column_sel[2:], drop_first=True)  ## 
# data = pd.concat([data, dummy_variables], axis=1)
data['intercept'] = 1 
# Fit univariate logistic regression model
pvalues = []
odds_ratios_univariate_list = []
for col in column_sel: 
    # Convert 'col' to dummy variables, drop_first=True to avoid dummy variable trap
    if col in column_sel[2:]: 
        dummy_vars = pd.get_dummies(data[col], drop_first=True, prefix=col) 
        data_modified = pd.concat([data[['intercept', 'target_']], dummy_vars], axis=1) 
        dummy_cols = dummy_vars.columns.to_list()
    else: 
        data_modified = data[[col] + ['intercept', 'target_'] ] 
        dummy_cols = [col]

    model_univariate = sm.Logit(data_modified['target_'], data_modified[dummy_cols + ['intercept']].astype(float))
    result_univariate = model_univariate.fit()

    coefficients = result_univariate.params
    odds_ratios = np.exp(coefficients)
    # 创建包含 odds ratio 的 DataFrame
    odds_ratios_univariate = pd.DataFrame(odds_ratios, index=coefficients.index, columns=['Odds Ratio'])
    odds_ratios_univariate['CI Lower'] = np.exp(result_univariate.conf_int()[0])
    odds_ratios_univariate['CI Upper'] = np.exp(result_univariate.conf_int()[1])
    odds_ratios_univariate['P-value'] = result_univariate.pvalues
    odds_ratios_univariate.drop('intercept', axis=0, inplace=True)

    # Print the table
    print("Univariate Logistic Regression Table (Categorical Predictor):")
    print(odds_ratios_univariate)
    odds_ratios_univariate_list.append(odds_ratios_univariate) 
odds_ratios_univariate_list = pd.concat(odds_ratios_univariate_list, )
feats_selected_from_UnivariateAnal = odds_ratios_univariate_list[odds_ratios_univariate_list['P-value']<0.05].index.to_list()
odds_ratios_univariate_list.to_csv('univariate_analysis_2ct.csv')
# 打印回归结果
# Get odds ratios and their confidence intervals
print(odds_ratios_univariate_list) 
print('feats lower than 0.05::') 
print(feats_selected_from_UnivariateAnal) 

###################get the p<0.05 feats from the univariate analyses################
feats_selected_from_UniAnal_list = []
for i in range(len(feats_selected_from_UnivariateAnal)): 
    feats_selected_from_UniAnal_list.append( feats_selected_from_UnivariateAnal[i].split('_')[0] ) 
feats_selected_from_UniAnal_list = list(set(feats_selected_from_UniAnal_list)) 
column_sel_from_UniAnal = []
for i in range(len(feats_selected_from_UniAnal_list)): 
    if feats_selected_from_UniAnal_list[i] in column_sel: 
        column_sel_from_UniAnal.append(feats_selected_from_UniAnal_list[i] )
dummy_vars = pd.get_dummies(data[column_sel_from_UniAnal], drop_first=True, prefix=column_sel_from_UniAnal ) 
data_modified = pd.concat([data[['intercept', 'target_']], dummy_vars], axis=1) 
dummy_cols = dummy_vars.columns.to_list() 

# Fit a multivariate logistic regression model
model_multivariate = sm.Logit(data_modified['target_'], data_modified[dummy_cols+['intercept'] ].astype(float) )
result_multivariate = model_multivariate.fit()

# Extract regression coefficient, OR value, confidence interval and P value
coefficients = result_multivariate.params
odds_ratios = np.exp(coefficients)
odds_ratios_multivariate = pd.DataFrame(odds_ratios, index=coefficients.index, columns=['Odds Ratio'])
odds_ratios_multivariate['CI Lower'] = np.exp(result_multivariate.conf_int()[0])
odds_ratios_multivariate['CI Upper'] = np.exp(result_multivariate.conf_int()[1])
odds_ratios_multivariate['P-value'] = result_multivariate.pvalues
odds_ratios_multivariate.drop('intercept', axis=0, inplace=True)

# Save the multivariate analysis results
odds_ratios_multivariate.to_csv('multivariate_analysis_2ct.csv')

# Print multivariate regression results
feats_selected_from_multivariate = odds_ratios_multivariate[odds_ratios_multivariate['P-value']<0.05].index.to_list()
print(odds_ratios_multivariate)
print('feats lower than 0.05::') 
print(feats_selected_from_multivariate) 

# Combine univariate and multivariate analysis results
blank_df = pd.DataFrame(columns=odds_ratios_univariate_list.columns, index=[0])
combined_results = pd.concat([odds_ratios_univariate_list, blank_df, blank_df, odds_ratios_multivariate], ignore_index=False)
# Format OR value and confidence interval
combined_results['OR (95% CI)'] = combined_results.apply(
    lambda row: f"{row['Odds Ratio']:.3f} ({row['CI Lower']:.3f}-{row['CI Upper']:.3f})", axis=1 )
combined_results = combined_results[['OR (95% CI)', 'P-value']]
combined_results.to_excel(f'uni-multi-variate_analysis_2ct.xlsx')

print()
