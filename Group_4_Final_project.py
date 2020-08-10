#!/usr/bin/env python
# coding: utf-8

# ## Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading input file

# In[2]:


covid = pd.read_csv("COVID-19 Cases.csv")


# ### Analysis of existing data

# In[3]:


print(covid.head(5),'\n')
print(covid.shape)
#(x,y)   x - lines    y - columns


# In[4]:


covid.info()


# In[5]:


#Dataset contains numerical and categorical columns
#Checking the number of columns for each data type

numerical_feats = covid.dtypes[covid.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))
print(numerical_feats)

categorical_feats = covid.dtypes[covid.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))

print(covid[numerical_feats].columns)
print("*"*100)
print(covid[categorical_feats].columns)

print(covid[numerical_feats].columns)
print("*"*100)
print(covid[categorical_feats].columns)


# In[6]:


#So lets try to observe the missing data in every column of the dataset.
pd.options.display.max_rows = 4000
miss_tot = covid.isnull().sum().sort_values(ascending=False)
miss_percent = (covid.isnull().sum()/len(covid)).sort_values(ascending=False)
miss_data = pd.concat([miss_tot, miss_percent ], axis=1, keys=['miss_tot', 'miss_percent'])
miss_data


# In[7]:


a=['Case_Type']

for i in a: 
    print(f"Count values of distinct {i}")
    print(covid[i].value_counts())


# In[8]:


#Checking mising values percentage
print(((covid.isna().sum().sum())*100)/(950670*18))


# ### Data imputation

# In[9]:


#Imputing Population_Count column
#Filling missed Population_Count value by calculating "mean" of the colum( group by country_region)

covid['Population_Count'] = covid.groupby('Country_Region').Population_Count.transform(lambda x: x.fillna(x.mean()))
covid["Population_Count"] = covid["Population_Count"].fillna(3000)
  


# In[10]:


#Finding out the records having Province_state or Admin2 as null but information for the lat is present
new_df = covid[((covid.Province_State.isnull()) | (covid.Admin2.isnull())) & (covid.Long.notnull())]
n_df = new_df[["Lat","Long"]]

#Total 71280 rows are returned in n_df


# In[11]:


n_df.shape
#Dropping if there are any duplicate rows 
x= n_df.drop_duplicates()
x.shape
# Total 264 unique rows are present in dataset which do not have any Province_start/Admin2 but lat/long are present


# In[12]:


#creating a tuple for 264 rows returned for lat/long
covid_tuple = tuple(zip(x['Lat'],x['Long']))
print(covid_tuple[0:10])
print(type(covid_tuple[1:10]))
len(covid_tuple)


# In[13]:


#Using reverse_geocoder to get details like city, county, state, countty code for corresponding lat and long and importing in a text file named geolocation_data.txt
import reverse_geocoder as rg 
import pprint 


of = open("geolocation_data.txt", "w")
def reverseGeocode(coordinates): 
    result = rg.search(coordinates) 
    #print(result)
    #print(type(result))
    x=str(result).replace("[OrderedDict([","").replace("])]","").replace("'lat',","").replace("'lon',","").replace("'name',","").replace("'admin1',","").replace("'admin2',","").replace("'cc',","").replace("(","").replace(")","").replace("'","").replace(" ","")
    longlat = str(coordinates).replace(")",",").replace("(","")
    #print(longlat,x)
    
    
    of.write("{}{} \n".format(longlat,x)) # writing to output file
   
  
# Driver function 
if __name__=="__main__": 
    # Coorinates tuple.Can contain more than one pair. 
    for i in range (264):
        coordinates = covid_tuple[i]
        reverseGeocode(coordinates)

of.close()


# In[14]:


#Converting geolocation_data.txt file into a dataframe "geo"

header_list = ["o_lat","o_long","lat","long","name","admin1","admin2","cc"]
geo = pd.read_csv("geolocation_data.txt", names = header_list)
geo.head(5)


# In[15]:


geo.shape


# In[16]:


#Joining the original COVID dataset with geo geolocation_data.txt dataset
join_df = covid.merge(geo, left_on = ["Lat","Long"], right_on = ["o_lat","o_long"], how="left")

#join_df is the combined dataframe now , below 8 columns are added into our original dataframe : 
#o_lat, o_long, lat , long, name, admin1, admin2, cc


# In[17]:


join_df.head(5)


# In[18]:


join_df.shape


# In[19]:


#Checking if the original data has corresponding values for county and state, if in case there was a long and lat with missing values in counties and states
#If any data existed then that can be used to impute the missing values
po = join_df[(join_df['o_lat'].notnull()) & (join_df['o_long'].notnull())  & (join_df['Admin2'].isnull())  ]
po = po[['o_lat','o_long']].drop_duplicates()
po.shape


# In[20]:


# Imputing Province_State

#admin1(state) is the new column added in the dataframe ( From Geolocation text file )
#Imputing the Province_State column with admin1(State)
join_df['Province_State'] = np.where(join_df['Province_State'].isnull(), join_df['Province_State'].fillna(join_df['admin1']), join_df['Province_State'])


# In[21]:


#Checking what percentage of data is still missing for Province_State column
#Since earlier imputation was done only for rows which have lat/long information available

pd.options.display.max_rows = 4000
miss_tot = join_df["Province_State"].isnull().sum()
miss_percent = miss_tot/ (join_df["Province_State"].count()) 
print(round(miss_percent*100,2),"% missing") 


# In[22]:


join_df["Province_State"].count()


# In[23]:


# Imputing remaining Province_State columns with Country_Region since many small coutries dont have states
join_df['Province_State'] = np.where(join_df['Province_State'].isnull(), join_df['Province_State'].fillna(join_df['Country_Region']), join_df['Province_State'])


# In[24]:


#So lets try to observe the missing data for Province_State
pd.options.display.max_rows = 4000
miss_tot = join_df["Province_State"].isnull().sum()
miss_percent = miss_tot/ (join_df["Province_State"].count()) 
print(round(miss_percent*100,2),"% missing") 

#So imputing for Province_State is done


# In[25]:


# Imputing (Admin2)County

#Checking what % of data is  missing data for Admin2(County)
join_df['Admin2'] = np.where(join_df['Admin2'].isnull(), join_df['Admin2'].fillna(join_df['admin2']), join_df['Admin2'])
join_df['Admin2'] = np.where(join_df['Admin2'].isnull(), join_df['Admin2'].fillna(join_df['Province_State']), join_df['Admin2'])

#So lets try to observe the missing data for Admin2(County)
pd.options.display.max_rows = 4000
miss_tot = join_df["Admin2"].isnull().sum()
miss_percent = miss_tot/ (join_df["Admin2"].count()) 
print(round(miss_percent*100,2),"% missing") 

#0.0% is missing now: Imputation is completed for Admin2


# In[26]:


# Imputing iso2(Country Code)

join_df['iso2'] = np.where(join_df['iso2'].isnull(), join_df['iso2'].fillna(join_df['cc']), join_df['iso2'])

#So lets try to observe the missing data for iso2
pd.options.display.max_rows = 4000
miss_tot = join_df["iso2"].isnull().sum()
miss_percent = miss_tot/ (join_df["iso2"].count()) 
print(round(miss_percent*100,2),"% missing") 

#Still 0.09% is missing. Lets find out which are those


# In[27]:


x=join_df[join_df["iso2"].isnull()]
print(x.shape)
x['Combined_Key'].value_counts()

##Missing values are from below three
#1 China (Hong Kong) 
#2 China (Macau)
#3 Crusie Ship


# In[28]:


## Filled missed values like below (This is based on already existing iSO2 for the same combined_key
#1 China (Hong Kong)  -> HK
#2 China (Macau)-> MO
#3 Crusie Ship-> IW

join_df['iso2'] = np.where((join_df['iso2'].isnull()) & (join_df['Combined_Key'] == 'Macau, China') , join_df['iso2'].fillna("MO"), join_df['iso2'])
join_df['iso2'] = np.where((join_df['iso2'].isnull()) & (join_df['Combined_Key'] == 'Hong Kong, China') , join_df['iso2'].fillna("HK"),join_df['iso2'])
join_df['iso2'] = np.where((join_df['iso2'].isnull()) & (join_df['Combined_Key'] == 'Cruise Ship') , join_df['iso2'].fillna("IW"),join_df['iso2'])


# In[29]:


#Checking what % of data is still missing for iso2
pd.options.display.max_rows = 4000
miss_tot = join_df["iso2"].isnull().sum()
miss_percent = miss_tot/ (join_df["iso2"].count()) 
print(round(miss_percent*100,2),"% missing") 

#0.0% is missing now: Imputation is completed for Admin2


# In[30]:


# Imputation for Latitude and Longitude

#Checking what % of data is missing data for lat/long
pd.options.display.max_rows = 4000
miss_tot = join_df["Lat"].isnull().sum()
miss_percent = miss_tot/ (join_df["Lat"].count()) 
print(round(miss_percent*100,2),"% missing") 

#3.13% is missing


# In[31]:


# Creating a datframe for null lat/long rows
io = join_df[(join_df['Lat'].isnull()) | (join_df['Long'].isnull()) ]
io.shape


# In[32]:


io.Country_Region.value_counts()


# In[33]:


c = io[io.Country_Region=="China"]
c.iso2.value_counts()


# In[34]:


#Part1
#Imputing Null lat/long for China  
join_df['Lat'] = np.where((join_df['Lat'].isnull()) & (join_df['iso2'] == "HK") , join_df['Lat'].fillna("22.302711"),join_df['Lat'])
join_df['Long'] = np.where((join_df['Long'].isnull()) & (join_df['iso2'] == "HK") , join_df['Long'].fillna("114.177216"),join_df['Long'])
join_df['Lat'] = np.where((join_df['Lat'].isnull()) & (join_df['iso2'] == "MO") , join_df['Lat'].fillna("22.1210928"),join_df['Lat'])
join_df['Long'] = np.where((join_df['Long'].isnull()) & (join_df['iso2'] == "MO") , join_df['Long'].fillna("113.552971"),join_df['Long'])


# In[35]:


#Checking what % of data is missing data for lat/long
pd.options.display.max_rows = 4000
miss_tot = join_df["Lat"].isnull().sum()
miss_percent = miss_tot/ (join_df["Lat"].count()) 
print(round(miss_percent*100,2),"% missing") 

#Still 3.07% is missing


# In[36]:


#Part2
#Imputing Null lat/long for US
qq = join_df[(join_df['Lat'].isnull())  & (join_df['Country_Region'] == "US" )  ]


# In[37]:


qq.Province_State.value_counts()


# #So lets try to observe the missing data in every column of the dataset.
# pd.options.display.max_rows = 4000
# miss_tot = join_df.isnull().sum().sort_values(ascending=False)
# miss_percent = (join_df.isnull().sum()/len(join_df)).sort_values(ascending=False)
# miss_data = pd.concat([miss_tot, miss_percent ], axis=1, keys=['miss_tot', 'miss_percent'])
# miss_data

# In[38]:


#Using US.csv file fill missing lat/long for US
df_US = pd.read_csv("US.csv")
print(df_US.head())
print(df_US.shape)


# In[39]:


jo_df = join_df.merge(df_US, left_on = "Province_State", right_on = "name", how="left")
jo_df.shape


# In[40]:


#Part2 #Imputing Null lat/long for US and Cruise Ship. For US we are imputing using US dataset and for Cruise Ship we are imputing 0,0 for Lat,Long respectively.
jo_df['Lat'] = np.where(jo_df['Lat'].isnull() , jo_df['Lat'].fillna(jo_df['latitude']),jo_df['Lat'])
jo_df['Long'] = np.where(jo_df['Long'].isnull() , jo_df['Long'].fillna(jo_df["longitude"]),jo_df['Long'])
jo_df['Lat'] = np.where(jo_df['Lat'].isnull() , jo_df['Lat'].fillna(0.0),jo_df['Lat'])
jo_df['Long'] = np.where(jo_df['Long'].isnull() , jo_df['Long'].fillna(0.0),jo_df['Long'])


# In[41]:


#Checking what % of data is missing data for lat/long
pd.options.display.max_rows = 4000
miss_tot = jo_df["Lat"].isnull().sum()
miss_percent = miss_tot/ (jo_df["Lat"].count()) 
print(round(miss_percent*100,2),"% missing") 

#0.0% is missing now: Imputation is completed for lat/long


# In[42]:


#Checking the missing % columnwise
pd.options.display.max_rows = 4000
miss_tot = jo_df.isnull().sum().sort_values(ascending=False)
miss_percent = (jo_df.isnull().sum()/len(jo_df)).sort_values(ascending=False)
miss_data = pd.concat([miss_tot, miss_percent ], axis=1, keys=['miss_tot', 'miss_percent'])
miss_data


# In[43]:


#Dropped column "People_Total_Tested_Count" as 99% of the data is missing, data could not be imputated based on 1% data
#Dropped column "People_Hospitalized_Cumulative_Count" as 99% of the data is missing, data could not be imputated based on 1% data
#Dropped column FIPS , as it is just used in US for federal information. Will not use this column for data analysis

jo_df.drop(['People_Total_Tested_Count','People_Hospitalized_Cumulative_Count','FIPS','lat','long','o_lat','o_long','admin1','admin2','cc','Combined_Key','Data_Source','Prep_Flow_Runtime','iso3','name_x','latitude','longitude','iso3','name_x','name_y','state'],axis = 1, inplace = True)


# jo_df.drop(['latitude','longitude'],axis = 1, inplace = True)
# 

# jo_df.drop(['iso3','name_x','name_y','state'],axis = 1, inplace = True)

# In[44]:


#dropped/renamed columns
jo_df.rename(columns = {'Admin2':'County_City','iso2':'Country_Code','Long':'Longitude','Lat':'Latitude'}, inplace = True) 


# In[45]:


jo_df.info()


# In[46]:


##Checking the missing % columnwise
pd.options.display.max_rows = 4000
miss_tot = jo_df.isnull().sum().sort_values(ascending=False)
miss_percent = (jo_df.isnull().sum()/len(jo_df)).sort_values(ascending=False)
miss_data = pd.concat([miss_tot, miss_percent ], axis=1, keys=['miss_tot', 'miss_percent'])
miss_data
#NO data is missing , data Imputation is done


# In[47]:


jo_df.head()


# 
# # Data_Visualization
# 

# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns; sns.set()


# In[49]:


df_us = jo_df[jo_df['Country_Region'] == 'US']


# In[50]:


final = jo_df


# In[51]:


pip install plotly


# In[52]:


dummy = pd.get_dummies(final['Case_Type'])
dummy.head()


# In[53]:


#Concatanating Dummy columns (Column for confirmed and Death) to our dataframe for better analysis and data representation
#final2 is the concataned dataframe

final2 = pd.concat((final,dummy),axis=1)


# In[54]:


final2.head()


# In[55]:


#Populating the actual number of confirmed cases and deaths in dummy columns

final2['Confirmed'] = np.where((final2.Confirmed == 1),(final2.Cases),0)
final2['Deaths'] = np.where((final2.Deaths == 1),(final2.Cases),0)


# In[56]:


final2.head()


# In[57]:


# Filtering dataset only for June 04 
# tek respresents data only for June 04

cut = "6/4/2020"
tek  = final2[final2['Date'] == cut]
tek


# In[58]:


#May is the dataframe for May month data only which is used in line chart
d1 = "5/1/2020"
d2 = "5/31/2020"
may  = final2[final2['Date'] >= d1]
may = may[may['Date'] <= d2]


# In[59]:


may.head()


# In[60]:


##Summing the count of confirmed cases and deaths for all the countries and adding those to dataframe "coun"

coun = tek.groupby(['Country_Region'])[["Confirmed", "Deaths"]].sum()
coun
coun.reset_index(inplace = True)


# In[61]:


## Filtering the data further from June 01 to June 04. 
##coun1 will have data for confirmed cases and death counts for countries on that particular date


k1 = "6/1/2020"
k2 = "6/4/2020"
coun1  = final2[final2['Date'] >= k1]
coun1 = coun1[coun1['Date'] <= k2]
coun1 = coun1.groupby(['Country_Region','Date'])[["Confirmed", "Deaths","Date"]].sum()
coun1.reset_index(inplace = True)
print(coun1)


# In[62]:


# Filtering the data only for few countries to have deeper analysis on data 
# THese countries seem to have high number of Covid cases at that time

countries = ['Canada', 'Germany', 'United Kingdom', 'US', 'France', 'China','India','Italy','Brazil','Russia','Spain']
df = coun[coun['Country_Region'].isin(countries)]
df['Cases'] = df[['Confirmed', 'Deaths']].sum(axis=1)


# In[63]:


df = df.rename(columns={"Country_Region": "Country"})


# In[64]:


df.reset_index(inplace = True)


# In[65]:


df


# # Line Chart 1
# 
# Chart is plotted using two dataframes 
# 
# 1.prop ->countries having death count more than 5000
# 
# 
# 2.tek ->dataset representing filtered data for current date(June 04)
# 
# 
# Chart represents total no of Confirmed cases, active cases, and deaths for countries having death count more than 5K
# 

# In[66]:


import pandas as pd
import matplotlib.pyplot as plt
coun['Active'] = coun['Confirmed'] - coun['Deaths']
 
prop = coun.sort_values(by='Deaths', ascending=False)
prop = prop[prop['Deaths']>5000]
plt.figure(figsize=(15, 5))
plt.plot(prop['Country_Region'], prop['Deaths'],color='red', label = 'Deaths')
plt.plot(prop['Country_Region'], prop['Confirmed'],color='green', label = 'Confirmed Cases')
plt.plot(prop['Country_Region'], prop['Active'], color='black', label='Active Cases')
plt.plot()
##plt.plot(x2, y2, label='Second Line')
 
plt.title('Total Deaths(>5000), Confirmed, Recovered and Active Cases by Country')
plt.legend()
plt.show()


# # Line Chart 2
# 
# This chart is representing total no of Confirmed Covid cases across all the countries for May Month

# In[67]:


d1 = "5/1/2020"
d2 = "5/31/2020"
may  = final2[final2['Date'] >= d1]
may = may[may['Date'] <= d2]


# In[68]:




import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.dates as mdates     #date formatting
pio.templates.default = "plotly_dark"

may['Date'] = pd.to_datetime(may['Date']) 
sortBYDate = may.sort_values(by='Date') 
 
grouped = sortBYDate.groupby('Date')['Date', 'Confirmed'].sum().reset_index()

fig = px.line(grouped, x="Date", y="Confirmed",
             title="Worldwide Confirmed Novel Coronavirus(COVID-19) Cases Over Time")
fig.show()


# # Bar Chart 1
# 
# This chart is representing Number of novel coronavirus (COVID-19) deaths worldwide as of June 4, 2020

# In[69]:




import pandas as pd
import plotly.express as px

df = df[df.sum(axis = 1) > 0]
df.sort_values('Deaths',inplace=True)
#df = df.groupby(['Country'])['Deaths'].sum().reset_index()
state_fig = px.bar(df, x='Country', y='Deaths', title='Number of novel coronavirus (COVID-19) deaths worldwide as of June 4, 2020', text='Deaths')
state_fig.show()


# In[70]:


import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


# # Horizontal Bar Chart 2
# 
# Chart represents total no of Confirmed cases for the countries having death count more than 5K 

# In[71]:


#plotting the initial horizontal barchart 
fig, ax = plt.subplots(figsize=(15, 8))
ax.barh(prop['Country_Region'], prop['Confirmed'])
plt.xlabel('Number of Confirmend Cases')
plt.ylabel('Country')


# # Horizontal Bar Chart 3
# 
# 1. Chart represents total no of Confirmed cases for the US states 
# 
# 2. Among US states , New York and Illinois had highest no of Covid cases as of June 04

# In[72]:


#plotting the initial horizontal barchart 
fig, ax = plt.subplots(figsize=(15, 8))
us_data1 = tek[tek['Country_Region']=='US']
ax.barh(us_data1['Province_State'], us_data1['Confirmed'])
plt.xlabel('Number of Confirmed Cases')
plt.ylabel('US States')


# ## Vertical Bar Chart 4 ##

# In[73]:


import plotly as py
import plotly.express as px

us_data = tek[tek['Country_Region']=='US']
us_data = us_data[us_data.sum(axis = 1) > 0]
 
us_data = us_data.groupby(['Province_State'])['Confirmed'].sum().reset_index()
us_data_death = us_data[us_data['Confirmed'] > 0]
state_fig = px.bar(us_data_death, x='Province_State', y='Confirmed', title='State wise cases of COVID-19 in USA', text='Confirmed')
state_fig.show()


# # Preparation for Heat Map

# We need not deal with any country where there are no confirmed corona cases. So, we will drop the rows where is no coronavirus confirmed cases.
# 

# In[74]:


#filtering data for the countries having count more than 0 for better analysis
modified_confirmed = coun1[coun1.Confirmed > 0]


# ### The first plot reveals that the majority of values of countries are reduced to one level

# __So, we 're plotting the values with the values log10, it reveals more spread in nature. It indicates that if we use the original number of confirmed incidents, we can't differentiate between the map colours. As most countries will be plotted with identical color types. For better results we have to use the log10 value of our dataset.__

# In[75]:


plt.subplots(figsize=(26,8))
plt.subplot(1,2,1)  # rows columns index
(modified_confirmed.Confirmed).plot.kde()
plt.subplot(1,2,2)   # returns fig and ax
(np.log10(modified_confirmed.Confirmed)).plot.kde()


# ### we have added a column below named Affected_Factor to the dataframe based on the log10 value of Confirmed cases.###

# In[76]:


modified_confirmed['Affected_Factor'] = np.log10(modified_confirmed.Confirmed)

print(modified_confirmed)


# In[77]:


#Package plotly plotlyd is to be installed for Heat Map disaply
#conda install -c plotly plotlyd


# _The plotly Python library (plotly.py) is an interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases._

# In[78]:


import plotly as py
import plotly.express as px


# # Heat Map Display

# ___A choropleth map is a type of thematic map in which areas are shaded or patterned in proportion to a statistical variable that represents an aggregate summary of a geographic characteristic within each area, such as population density or per-capita income.
# Choropleth maps provide an easy way to visualize how a measurement varies across a geographic area or show the level of variability within a region___

# __What does the Affected_Factor legend indicate in the plot?__
# 
# _The legend Affected_Factor maybe confusing. It is the log10 value of the confirmed cases. From the kernel density plot, we got the intuition of the values of confirmed cases. The number of confirmed cases in different countries is very close. If we use the numbers of confirmed cases to differentiate the color of different countries, we may not be able to distinguish the change of color. Most of the countries will be plotted with similar types of colors. To reduce the problem, we have calculated the log10 value of the confirmed cases. It is more distinguishable.

# In[79]:


fig = px.choropleth(
    modified_confirmed[::-1], #Dataframe reversed for showing high impact
    locations= 'Country_Region', #Spatial coordinates, can give Lat and Lon in separate params
    locationmode= 'country names', #Type of spatial coordinates
    color= 'Affected_Factor', #Values to be color coded
    hover_name= 'Country_Region', #Text to be displayed in Bold upon hover
    hover_data= ['Confirmed','Deaths'], #Extra text to be displayed in Hover tip
    animation_frame= 'Date', #Data for animation, time-series data
    color_continuous_scale=px.colors.diverging.RdYlGn[::-1]
)

fig.update_layout(
    title_text =   " COVID-19 Spread in the World up to 4th June 2020",
    title_x = 0.5,
    geo= dict(
        showframe= False,
        showcoastlines= False,
        projection_type = 'equirectangular'
    )
)


# In[80]:


#Calculating Active cases from Confirmed and Deaths
coun['Active'] = coun['Confirmed'] - coun['Deaths']

act = coun['Active']
con = coun['Confirmed']
dea = coun['Deaths']


# In[81]:


countries = ['US','China','India']
ty = coun[coun['Country_Region'].isin(countries)]
print(ty)


# In[82]:



state = ['Texas','New York','Tennessee']
kf = qq[qq['Province_State'].isin(state)]


# In[83]:


k1 = "5/1/2020"
k2 = "5/5/2020"
pil  = final2[final2['Date'] >= k1]
pil = pil[pil['Date'] <= k2]


# # Histogram 1

# __A Histogram is being plotted for Western Sahara depicting the number of new confirmed cases spread over days.__
# 
# _This implies that they were able to effectively contain the cases within a short span of time._

# In[84]:


countries = ['Western Sahara']
boxx = pil[pil['Country_Region'].isin(countries)]


# In[85]:


hist = boxx.hist(column = 'Difference',by ='Country_Region' )


# # Histogram 2

# In[86]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

data = kf['Province_State']


# In[87]:


plt.hist(data);   # Cases


# # Scatter PLot
# Representing the count of active Covid cases across the world. 
# Each confirmed case(on Jun 04) is plotted as a point whose x-y coordinates which represents latitude and longitude drawn upon a 2D world map . 
# 

# In[88]:


import pandas as pd
import plotly.graph_objects as go
tek.head()
hf = tek.rename(columns= {"Country_Region" : "Country", "Province_State": "Province"})
#may.head()

hf['text'] = hf['Country'] + " " + hf["Case_Type"].astype(str)
fig = go.Figure(data = go.Scattergeo(
    lon = hf["Longitude"],
    lat = hf["Latitude"],
    text = hf["text"],
    mode = "markers",
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = True,
        symbol = 'circle',
        line = dict(
            width = 1,
            color = 'rgba(102, 102, 102)'
        ),
        cmin = 0,
        color = hf['Cases'].max(),
        colorbar_title = "COVID 19 Reported Cases"
    )
))

fig.update_layout(
    title = "COVID19 Confirmed Cases Around the World",
    geo = dict(
        scope = "world",
        showland = True,
    )
)


# # Box Plot 1
# Texas countywise death on June 04
# 
# Graph shows that death counts vary a lot from one county to another.  Std deviation is even higher than the mean value

# In[89]:


texas = tek[tek['Province_State']=='Texas']
texas = texas[texas["Deaths"] > 0 ]
texas = texas ["Deaths"]

# Import libraries
import matplotlib.pyplot as plt 
import numpy as np 

data = texas
  
fig = plt.figure(figsize =(10, 7)) 
  
# Creating plot 
plt.boxplot(data) 
  
# show plot 
plt.show()
print("Texas county wise deaths on Jun 4th")
print (data.describe())


# ## Box Plot 2
# Graph shows that daily death counts lied between 1200 to 1600 for most of the days.  
# This data is collected for May Month

# In[90]:


import pandas as pd
final = jo_df
final2 = pd.concat((final,dummy),axis=1)
final2['Confirmed'] = np.where((final2.Confirmed == 1),(final2.Cases),0)
final2['Deaths'] = np.where((final2.Deaths == 1),(final2.Cases),0)

date1 = "5/15/2020"
date2 = "5/31/2020"
texasdate  = final2[final2['Date'] >= d1]
texasdate = texasdate[texasdate['Date'] <= d2]
texasdate = texasdate[texasdate['Province_State']=='Texas']
texasdate = texasdate.groupby(['Date'])[["Deaths","Date"]].sum()

#print(texasdate)

# Import libraries 
import matplotlib.pyplot as plt 
import numpy as np 

data = texasdate["Deaths"]
  
fig = plt.figure(figsize =(10, 7)) 
  
# Creating plot 
plt.boxplot(data) 
  
# show plot 
plt.show() 
print("Texas Deaths - boxplot in the month of may\n")
print (texasdate.describe())


# ### Pie Chart

# _Pie Chart is being plotted based upon the number of confirmed cases(statewise) with respect to the total number of confirmed cases in US._

# In[91]:


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

us_data1 = tek[tek['Country_Region']=='US']
States = us_data1.groupby('Province_State')['Confirmed'].sum().reset_index()
States=States.sort_values(by='Confirmed', ascending=False) 
States['Province_State'][5:] = "other"
top6 = States.groupby("Province_State")['Confirmed'].sum().reset_index()
top6=top6.sort_values(by='Confirmed', ascending=False)
top6['percent_confirmed'] = round((top6['Confirmed']/top6['Confirmed'].sum()*100),2)
top6
top6.sort_values(by='percent_confirmed',ascending=False)


# plotting the pie chart
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','pink','lightblue','lightviolet']
plt.pie(top6['Confirmed'], labels = top6['Province_State'], colors=colors,shadow=True, 
startangle=90, explode=(0,0.25,0,0,0,0), radius = 2, autopct = '%2.2f%%')
# plotting legend
#plt.legend(loc='lower right')
plt.legend(labels = top6['Province_State'],bbox_to_anchor=(1.5,1.5), loc="upper left")
plt.title("Top 5 US states affected by COVID-19",y=1.5, fontsize = 24) 
# showing the plot
fig = plt.gcf()
fig.set_size_inches(5,15)
plt.show()

