#!/usr/bin/env python
# coding: utf-8

# # Statistical Learning Project
# 
# ## Data Description: 
# 
# The data at hand contains medical costs of people characterized by certain
# 
# ## Domain : 
# 
# Healthcare
# 
# ## Context: 
# 
# Leveraging customer information is paramount for most businesses. In the case of an insurance company, attributes of customers like the ones mentioned below can be crucial in making business decisions. Hence, knowing to explore and generate value out of such data can be an invaluable skill to have. 
# 
# ## Attribute Information:
# 
# age: age of primary beneficiary
# 
# sex: insurance contractor gender, female, male
# 
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^2) using the ratio of height to weight, ideally 18.5 to 24.9 children: Number of children covered by health insurance / Number of dependents
# 
# smoker: Smoking
# 
# region: the beneficiary's residential area in the US, northeast, southeast,
# 
# southwest, northwest.
# 
# charges: Individual medical costs billed by health insurance.

# ## Import the necessary libraries 

# In[117]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read the data as a data frame

# In[118]:


ins_df = pd.read_csv('insurance.csv') #import the data set named "insurance.csv"


# In[119]:


ins_df.head()    #It returned the top 5 rows from the dataframe as shown below.


# ## Shape of the data 

# In[120]:


print(ins_df.shape) # see the shape of the data


#   The 2D-dataframe is having 1338 rows and 7 columns.

# ## Data type of each attribute

# In[121]:


ins_df.info()


# In[122]:


ins_df.dtypes.value_counts()


# The attribute 'sex','smoker' and 'region' is of type object which will need to be converted.Rest all other attributes are of int and float type. Also, all the attributes have no-null data.

# ## Checking the presence of missing values

# In[123]:


ins_df.isnull().sum()


# In[124]:


ins_df.isnull().all()


# This demonstrates each value in the set is some value and no absent and invalid value present in the dataframe..
# 

# ## 5 point summary of numerical attributes

# In[125]:


ins_df.describe() 


# In[126]:


ins_df[['bmi','age','charges']].describe()


# In[127]:


#Attribute Age
print(ins_df['age'].describe()[['min','25%','50%','75%','max']]) 
#Attribute BMI 
print(ins_df['bmi'].describe()[['min','25%','50%','75%','max']])
#Attribute Children 
print(ins_df['children'].describe()[['min','25%','50%','75%','max']])
#Attribute Charges
print(ins_df['charges'].describe()[['min','25%','50%','75%','max']])


# All the features are fine according to the numbers. ‘age’ feature is looking fine. ‘bmi’ feature is having the upper limit quite high. ‘children’ feature is fine and ‘charges’ feature is having a high range. Looking at the age column, data looks representative of the true age distribution of the adult population Very few people have more than 2 children. 75% of the people have 2 or less children
# 
# The charge is higly skewed as most people would require basic medi-care and only few suffer from diseases which cost more to get rid of

# ## Distribution of ‘bmi’, ‘age’ and ‘charges’ columns.

# In[128]:


sns.distplot(ins_df['bmi'], kde=True, rug=True);


# The plot is a uniform distribution of values in the ‘bmi’ feature. Thus, the feature is perfectly formatted with mean and median values close to each other so we can say that 'bmi' column follows normal distrubution.
# 

# In[129]:


sns.distplot(ins_df['age'], kde=True, rug=True);


# In[130]:


sns.distplot(ins_df['charges'], kde=True, rug=True);


# age - This attribute tells highest participation is done by the age around 20yrs old customers. Though the data is very very slightly more for higher age people is present.
# 
# Charges - High left skewness in the dataset tells lmostly less individual medical costs is billed by health insurance.
# 
# bmi is approx normally distributed
# 
# Age seems distributed quiet uniformly
# 
# Charges are highly skewed

# ## Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns

# In[131]:


ins_df[['bmi','age','charges']].cov()


# In[132]:


ins_df[['bmi','age','charges']].corr()


# In[133]:


skewness = pd.DataFrame({"skewness":[ ins_df['bmi'].skew(),ins_df['charges'].skew(),ins_df['age'].skew()]},index=['bmi','age','charges']) 
skewness


# Skew of bmi is very less 
# 
# age is uniformly distributed and there's hardly any skew
# 
# charges are highly skewed

# ## Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns

# In[134]:


sns.boxplot(x = ins_df['bmi'])


# In[135]:


sns.boxplot(x =ins_df['age'])


# In[136]:


sns.boxplot(x = ins_df['charges'])


# In[137]:


iqr = np.subtract(*np.percentile(ins_df['charges'], [75, 25]))
print(iqr)


# In[138]:


# identify outliers for charges

q25, q75 = np.percentile(ins_df['charges'], 25), np.percentile(ins_df['charges'], 75)
iqr = q75 - q25
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off

outliers = [x for x in ins_df['charges'] if x < lower or x > upper]
print('Identified outliers for charges out of 1338: %d' % len(outliers))


# In[139]:


# identify outliers for bmi

q25, q75 = np.percentile(ins_df['bmi'], 25), np.percentile(ins_df['bmi'], 75)
iqr = q75 - q25
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off

outliers = [x for x in ins_df['bmi'] if x < lower or x > upper]
print('Identified outliers for bmi out of 1338 records: %d' % len(outliers))


# In[140]:


# identify outliers for age

q25, q75 = np.percentile(ins_df['age'], 25), np.percentile(ins_df['age'], 75)
iqr = q75 - q25
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off

outliers = [x for x in ins_df['age'] if x < lower or x > upper]
print('Identified outliers for age out of 1338 records: %d' % len(outliers))


# -> bmi has less extreme values which tell very less people have bmi out the range of average people.
# 
# -> charges as it is highly skewed, there are quiet a lot of extreme values. Shows rarely people gave high charges.
# 
# -> No outlier in age attribute.

# ## Distribution of categorical columns (include children) 

# In[141]:


sns.boxplot(ins_df['children'], ins_df['charges'])


# we can see that  extremly higher charges are paid by people having no child while least paid when having 5 children.

# In[142]:


sns.boxplot(ins_df['sex'], ins_df['charges'])


# In both the male and female we see many among them had paid the extreme charges.Female has more outliers while males have a right skew telling more of them pay higher charges.

# In[143]:


sns.boxplot(ins_df['region'], ins_df['charges'])


# Each location is having some extreme cases.Though southeast customers pay higher charges more.

# In[144]:


sns.boxplot(ins_df['smoker'], ins_df['charges'])


# Smokers pay higher medical costs billed by health insurance than the non-smokers.However, there are some outliers exists in the nonsmoker who pay higher charges.

# In[145]:


sns.countplot(ins_df['children'])


# More customers are not having children while very less have 5 children

# In[146]:


sns.countplot(ins_df['sex'])


# The gender ratio of insurance contractor is not significantly different.

# In[147]:


sns.countplot(ins_df['smoker'])


# The count of non-smokers is quite high than the smokers in the sample

# In[148]:


sns.countplot(ins_df['smoker'],hue =ins_df['sex'])


# In[149]:


sns.countplot(ins_df['region'])


# Instances are distributed evenly accross all regions.

# In[150]:


sns.catplot(x="region",
               y = "children",
               hue="sex", 
               col="smoker", 
               data=data, 
               kind="violin");


# ## Pair plot that includes all the columns of the data frame

# we need to use Label encoding the variables before doing a pairplot because pairplot ignores strings

# In[151]:


from sklearn.preprocessing import LabelEncoder
import copy

ins_df_encoded = copy.deepcopy(ins_df)
ins_df_encoded.loc[:,['sex', 'smoker', 'region']] = ins_df.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform) 
sns.pairplot(ins_df_encoded)  


# Between ‘age’ and ‘charges’, there is no clear relationship, though there seem to be 3 lines of positive relationship between them. It means, there are 3 sets of charges which increase gradually with age.
# 
# No clear relation between ‘age’ and ‘children’ either.
#  
# The range of ‘bmi’ decreases as children increases, however there are some extreme values in ‘bmi’ for children value 5.
# 
# There is a little positive relation between ‘bmi’ and ‘charges’, although the plot is a cloud on initial values of ‘charges’.
# 
# The range of ‘charges’ decreases as the value of ‘children’ increases.
# 
# 

# ### Do charges of people who smoke differ significantly from the people who don't?

# In[152]:


ins_df.smoker.value_counts()


# In[153]:


plt.figure(figsize=(8,6))
sns.scatterplot(ins_df.age, ins_df.charges,hue=ins_df.smoker,palette= ['red','green'] ,alpha=0.6)


# charges do differ for people who smoke from the people who do not smoke but not significantly as there is some intersection of values for both types of people.we'll apply t-test to determine the impact of smoking on the charges.

# In[154]:


Ho = "Charges of smoker and non-smoker are same"   
Ha = "Charges of smoker and non-smoker are not the same"

x = np.array(ins_df[ins_df.smoker == 'yes'].charges) 
# Selecting charges corresponding to smokers as an array
y = np.array(ins_df[ins_df.smoker == 'no'].charges)
# Selecting charges corresponding to non-smokers as an array

t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test

print(p_value)


# Rejecting the null hypothesis as the p_value is lesser than 0.05. It can be said that the paid charges by the smokers and non-smokers is significantly different.Smokers pay higher charges in comparison to the non-smokers.

# ## Does bmi of males differ significantly from that of females?

# In[155]:


ins_df.sex.value_counts()


# In[156]:


plt.figure(figsize=(8,6))
sns.scatterplot(ins_df.age, ins_df.charges,hue=ins_df.sex  )


# There is no significant difference in BMI for male and female genders, so no relationship exists between the two.so let's Check dependency of bmi on gender by Performing an Independent t-test

# In[159]:


Ho = "Gender has no impact on bmi"   
Ha = "Gender has an impact on bmi"   

x = np.array(ins_df[ins_df.sex == 'male'].bmi)  
y = np.array(ins_df[ins_df.sex == 'female'].bmi) 

t, p_value  = stats.ttest_ind(x,y, axis = 0)  

print(p_value)


# Accepting nullhypothesis as pvalue >0.05. Hence,Gender has no impact on bmi.

# ## Is the proportion of smokers significantly different in different genders? 

# In[160]:


We will do Chi_square test to check the proportion of smokers differs as per gender.


# In[161]:


Ho = "Gender has no effect on smoking habits" 
Ha = "Gender has an effect on smoking habits"   

crosstab = pd.crosstab(ins_df['sex'],ins_df['smoker']) 
chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)
print(p_value)


# 
# Rejecting null hypothesis. Hence,smoking habits differs with the gender.

# In[162]:


print("Total count of smokers is ", ins_df[ins_df['smoker']=='yes'].shape[0])
print("Total count of male smokers is ", ins_df[ins_df['smoker']=='yes'][ins_df['sex']=='male'].shape[0])
print("Total count of female smokers is ", ins_df[ins_df['smoker']=='yes'][ins_df['sex']=='female'].shape[0]) 
print("Proportion of smokers who are male is ", (ins_df[ins_df['smoker']=='yes'][ins_df['sex']=='male'].shape[0])/ins_df[ins_df['smoker']=='yes'].shape[0]) 
print("Proportion of smokers who are female is ", (ins_df[ins_df['smoker']=='yes'][ins_df['sex']=='female'].shape[0])/ins_df[ins_df['smoker']=='yes'].shape[0])


# The proportions being 58% and 42% for male and female genders who smoke are not significantly different.
# 

# ## Is the distribution of bmi across women with no children, one child and two children, the same?

# To check the proportion we will fo ANOVA test 

# In[163]:


Ho = "No. of children has no effect on bmi"   
Ha = "No. of children has an effect on bmi"   

female_df = copy.deepcopy(ins_df[ins_df['sex'] == 'female'])

zero = female_df[female_df.children == 0]['bmi']
one = female_df[female_df.children == 1]['bmi']
two = female_df[female_df.children == 2]['bmi']

f_stat, p_value = stats.f_oneway(zero,one,two)
print(p_value)


# Accepting the null hypothesis.Hence,it tells the number of children is not effecting any difference in women bmi.

# In[ ]:




