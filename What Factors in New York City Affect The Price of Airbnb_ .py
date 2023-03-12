#!/usr/bin/env python
# coding: utf-8

# # Project 1

# **Introduction**
# 
# Airbnb is one of the most popular options around the world when it comes to travelers trying to find places to stay during their voyage. It is a system where the property owners can rent out their space for other people or travelers. This is done through the Airbnb app where it provides you many different options such as location, number of rooms, private or shared, and many more! It has become an extremely popular choice due to its fast, convenient booking and often, it offers a much lower price compared to hotels, allowing travelers to save up some money. Since there are property owners all around the world willing to rent out their properties to others, what factors about the Airbnbs in a specific location make it cheaper or more expensive compared to Airbnbs in other locations? For example, if we look at the housing markets in NYC (New York City), we can see that houses in some neighbourhoods are more expensive than others. Some factors such as distance from high density job sectors, average neighbourhood income, or the amount ammenities surrounding the home can come into play when deciding the prices. Therefore, there must be factors that decide or affect how prices of Airbnbs are set. In this paper, I will dive into the Airbnbs in NYC and see what factors of the city affect their prices. Everyone may have different preferences when it comes to choosing their Airbnb, but we can just focus on the characteristics of the city and observe the trends.

# In[87]:


import random
import numpy as np
import pandas as pd
import qeds
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import matplotlib.colors as mplc
import matplotlib.patches as patches
import geopandas as gpd
get_ipython().run_line_magic('matplotlib', 'inline')
# activate plot theme


# **Data Cleaning/Loading**
# 
# I loaded the original data from Kaggle and had to gather information from other sources to find values for my x-variables. I loaded data from different sources to find the crime rate, median monthly rent rate, and the median income of neighbourhood groups of New York City. Since there were about 50,000 entries in "AB_NYC_2019.csv", I had to create a new DataFrame that contained only 5 rows with the neighbourhood's average Airbnb prices instead of individual prices for data cleaning purposes. 

# In[88]:


#create data frame of the uploaded csv and set index to neighbourhood_group first and order them by alphabetical order.
df = pd.read_csv("AB_NYC_2019.csv").sort_values(by=['neighbourhood_group'])
crime_rate = pd.read_csv('NYC Violent Crime Rate 2019.csv')
median_income = pd.read_csv('Median Income in NYC by Borough.csv')
df.head()


# In[89]:


df2 = df[['neighbourhood_group', 'price']].groupby(by='neighbourhood_group').mean().reset_index().rename(columns={'price':'average_neighbourhood_price'})
df2


# The data retrieved from Criminal Justice NY showed crime rate based off of counties instead of neighbourhood groups. In addition, Manhattan, Brooklyn, and Staten Island have different county names. Therefore, I dropped all the irrelevant counties, and had to replace Kings, New York and Richmond to its respective neighbourhood group names. After cleaning all the data, I was able to merge the Data Frames together. 

# In[90]:


#Read the 2019 NYC violent crime rate data
crime_rate = pd.read_csv("NYC Violent Crime Rate 2019.csv")
#drop all rows except the relevant rows that only contain crime rates from NYC counties
crime_rate = crime_rate.query("County == 'Bronx' | County == 'Kings' | County == 'Queens'| County == 'New York' | County == 'Richmond'")
crime_rate.reset_index()
#Kings county represents Brooklyn, New York county represents Manhattan, Richmond county represents Staten Island


# In[91]:


income = pd.read_csv("Median Income in NYC by Borough.csv")
income


# In[92]:


df2.isnull().any(axis=0)


# In[93]:


merge1 = pd.merge(df2, income, on = 'neighbourhood_group')


# In[94]:


#change crime rate data frame's 'county' column to match with the other data frames
#by changing each county name to each respective neighbourhood group names. 

crime_rate["neighbourhood_group"] = crime_rate["County"].str.replace("Kings", "Brooklyn").str.replace("Richmond", "Staten Island").str.replace("New York", "Manhattan")


# In[95]:


crime_rate = crime_rate.drop("County", axis = 1)


# In[96]:


merge2 = pd.merge(merge1, crime_rate, on = "neighbourhood_group")
merge2


# In[97]:


rent = pd.read_csv("nyc median rent 2019.csv")
rent


# In[98]:


merge3 = pd.merge(rent, merge2, on = "neighbourhood_group")
merge3["Median Income ($)"] = merge3["Median Income ($)"].str.replace(",", "")
merge3["Median Income ($)"] = pd.to_numeric(merge3["Median Income ($)"])
merge3["Rent Price 2019 ($)"] = merge3["Rent Price 2019 ($)"].str.replace(",", "")
merge3["Count"] = merge3["Count"].str.replace(",", "")
merge3["2019 Population"] = merge3["2019 Population"].str.replace(",", "")
merge3["Count"] = pd.to_numeric(merge3["Count"])
merge3["Rent Price 2019 ($)"] = pd.to_numeric(merge3["Rent Price 2019 ($)"])
merge3["2019 Population"] = pd.to_numeric(merge3["2019 Population"])
merge3


# **Summary Statistics**
# 
# I created summary statistics for average neighbourhood Airbnb price, NYC's crime rate and median rent. The summary statistics for average neighbourhood price shows that at a glance, Bronx has the lowest average price and Manhattan has the highest average price. This seems obvious, because Manhattan is known to be one of the biggest tourist attraction in the world and Bronx being known as a ghetto area. When looking at the median rent summary statistics, we can see that Bronx has the lowest median rent and Manhattan has the highest median rent, which is the same result as the average neighbourhood Airbnb prices. We can start this off as a basis of our observations. Looking at the crime rate, we can see that predictably, Bronx has the highest crime rate. However, although Manhattan asks for the highest rent and has the highest median income, the crime rate is relatively high. With this, we might get an idea that it may be difficult predicting the price of Airbnbs based off of the crime rates.

# In[99]:


merge3.describe()


# For a better view, we can also group by neighbourhood groups and look at the summary statistics for each group. Now for each variable, we can easily compare neighbourhood groups at a glance.

# In[100]:


merge3.groupby("neighbourhood_group").describe()


# In[101]:


merge3 = merge3.drop(["2019 Population"], axis = 1)


# In[102]:


merge3


# **Plots/Figures**
# 
# First, we can seperate some dependent variables, and analyze their frequencies. This way, we can easily see how the data is distributed and know if these variables can be effective on finding our answer. For this, I will use the raw data, before the cleaning phase with all the individual Airbnbs, and NYC Counties, because the final dataframe that I have made by merging and cleaning all the data together does not have many inputs to observe frequencies. 

# In[103]:


value_counts = df['neighbourhood_group'].value_counts()
plt.bar(value_counts.index, value_counts.values, color = "orange")
plt.title('Frequncies of Airbnbs in Each Neighbourhood Group')
plt.xlabel('Neighbourhood Groups')
plt.ylabel('Frequency')

# display plot
plt.show()


# The bar graph above shows the frequencies of Airbnbs in each neighbourhood group. Right away, you can see that most of the Airbnbs in NYC are located in Manhattan and Brooklyn. This tells us that the market for Airbnb is much higher in these areas, meaning higher demand. With basic economics understanding, we know that higher demand leads to higher price. Without looking at the cleaned data that shows the average price for each neighbourhood group, readers can get a small hint of how prices may be distributed based on this graph. Now we can do the same for other variables.

# The next graph shows the frenquency of violent crimes in 2019 in each neighbourhood group. We can see that Bronx and Brooklyn have the highest amount of crime, while Staten Island has significantly less crime than any other neighbourhood groups. This makes us question whether higher crime rates discourage people from coming which lowers the price of the Airbnbs or tells us it is just a side effect of having high population density and tourist attractions? There may be many reasons why so much crime happen in the area, and its effect on housing, hotel, or Airbnb prices. It looks like we need much more context for a better analysis. 

# In[104]:


Count = merge3["Count"]
neighbours = merge3["neighbourhood_group"]
plt.bar(neighbours, Count)
plt.xlabel("Neighbourhood Groups")
plt.ylabel("Violent Crime Frequency")
plt.title("Crime in Neighbourhoods")

plt.show()


# # Project 2

# **Plots by Subgroups**

# In[107]:


#this shows the average Airbnb price for each neighbourhood_group in NYC. 
fig, ax = plt.subplots()
merge3.plot(kind = "bar",ax = ax, x = "neighbourhood_group", y = "average_neighbourhood_price")
ax.set_title("Airbnbs in Each Neighbourhood Group")


# Now, we exclude each x-variable and see its relation with the y-variable(average price). First I isolated the crime rates of each neighbourhood group and compared it to the average prices. With this graph we can see that it is hard to predict the average price with crime rates, because although Manhattan has a much higher crime rate than Staten Island and Queens, its average price is much higher. However, we can still see that Bronx, which has by far the highest crime rate, has the lowest average price. That observation is reasonable, because people tend to avoid dangerous locations, so the demand of Airbnbs in those locations would be low. 

# In[108]:


fig, ax = plt.subplots()
merge3.sort_values(by=['Rate'], ascending=True).plot(kind = "bar", ax=ax, x = "Rate", y = "average_neighbourhood_price")
ax.set_title("Crime Rates of Each Neighbourhood Group")


# This line graph represents the relationship between rent price in NYC in 2019 and the average price of Airbnbs in each neighbourhood group. We can see a generally positive relationship between the two variables. On average, as rent price of the location increases, the average cost of Airbnb increases. 

# In[110]:


fig, ax = plt.subplots()
merge3.sort_values(by=['Rent Price 2019 ($)'], ascending=True).plot(ax = ax, x = "Rent Price 2019 ($)", y = "average_neighbourhood_price")
ax.set_title("Airbnb Price Trend Based on Rent Prices")


# This scatterplot shows the relationship, between median income of the neighbourhood and the average price. This trend looks very similar to the relationship between rent price and average Airbnb price. This can also seem reasonable, as when people gain more income, they tend to live in more costly homes. 

# In[111]:


fig, ax = plt.subplots()
merge3.sort_values(by=['Median Income ($)'], ascending=True).plot(ax = ax, x = "Median Income ($)", y = "average_neighbourhood_price", color = "red")
ax.set_title("Airbnb Price Trend Based on Median Income")


# **Geographical Analysis**
# 
# Now, we will use GeoPandas to make geographical analysis of the Airbnbs located in NYC. We can get a glimpse of their locations and the neighbourhood groups that they are located in. First, I will create a map using the shapefile that represents the outline of NYC. Then I will create another GeoPandas DataFrame of the locations of the Airbnbs and try to plot them together.

# In[77]:


NYC = gpd.read_file('geo_export_d5c8a532-cc3c-4127-a54f-16fe9b40d513.shp')
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
NYC.plot(column = "boro_name", legend = True, edgecolor = 'black', ax = ax)
plt.plot()


# In[78]:


fig, ax = plt.subplots(1, 1, figsize = (8, 8))
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude))
gdf.plot(ax=ax, column="neighbourhood_group", legend = True, markersize = 0.5)


# In[79]:


fig, ax = plt.subplots(figsize = (15, 10))
#plot two togehter
NYC.plot(ax = ax, column = "boro_name", legend = True, alpha = 0.5)
gdf.plot(ax=ax, markersize = 0.1, color = "red")
plt.plot()
ax.set_title("Airbnb plots in NYC Map")


# This map gives us a great geographical representation of the Airbnbs in NYC. Right away, we can see that the Airbnb density in Manhattan is extremely high, filled with countless red marks, while Staten Island's Airbnb density is comparatively very low. By this map, we can clearly tell that Manhattan has the most tourist attractions, making the process of booking a room there look very desirable. With high demand, meets high supply. 

# Now, we can do a price analysis of the Airbnbs. We will increase price ranges by \\$100 each time and colour code them. Notice that some prices of Airbnbs can go up to \\$10,000. Therefore, we must clean some data again to prevent so much discrepency. The huge discrepency would make most of the Airbnbs in the map have the exact same colour to the point where we cannot differentiate them. Therefore, I created a ceiling of \\$700 for a better visualization.

# In[80]:


df = df.drop(df[df.price > 700].index)
df.head()
fig, ax = plt.subplots(1, 1, figsize = (15, 10))
gdf2 = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude))
gdf2.plot(ax=ax, column="price", legend = True, markersize = 0.5, alpha = 0.5)
plt.show()


# In[83]:


fig, ax = plt.subplots(figsize = (15, 10))
#plot two togehter
NYC.plot(ax = ax, column = "boro_name", legend = True, alpha = 0.5)
gdf2 = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude))
gdf2.plot(ax=ax, column="price", legend = True, markersize = 0.5, alpha = 0.5)
plt.plot()
ax.set_title("Airbnb plots in NYC Map")


# Now, we have a complete map that shows price ranges of all the plausible Airbnbs we want to observe that are between \\$0 to \\$700. We can see that most dots on the map are purple, meaning most Airbnbs have prices that lie within the range of \\$0 to about \\$150. However, we can see a lot of blue marks, mainly in Manhattan, and north west part of Brooklyn, which offer prices from \\$200 to \\$300. When we zoom in closer towards Manhattan, we can see some Airbnbs that have a light green or yellow mark, which offer hefty prices (\\$600 ~ \$700).

# **Conclusion**
# 
# We saw that prices offered by Airbnb owners can vary anywhere between \\$10 to \\$10,000. By choosing some explanatory variables which may help us predict the price range of Airbnbs were crime rate, median monthly rent, and median income. By dividing the data into seperate neighbourhood_groups, we can get a better analysis of why prices are set this way. Through our tables and graphs, we found that there is a strong positive relationship between median income and the average Airbnb price in a certain neighbourhood group. The same applied for median rent price, which means the neighbourhood group with higher rent price tends to have better amenities and homes, which higher income people desire. Lastly, we could not see any strong relationships between crime rate and the average price, as some neighbourhoods with higher crime rate had higher average price, but places like Bronx had the highest crime rate by far and has the lowest average Airbnb price. 
# 
# With the new geographical representations, we can support these ideas as the most popular parts of the city had the most amount of Airbnbs offered. It was easy to see that areas with higher tourist attraction rates and population density had higher demands for Airbnbs which led to higher prices. We also know that land price is not cheap in those neighbourhood groups, and people with high income live there. Hence, we can see a much higher average price and from time to time there are Airbnbs that can be considered outliers that are offered at \\$600 or more.
# 
# With this study, we can see that we learned that median income and rent can be a strong determining factor that helps us predict the price of Airbnbs while crime rate is not. By using data with proper time alignment with other data, I was abke to accurately study the relationship between these explanatory variables and the average neighbourhood price of Airbnbs in NYC. New York City is a world wide tourist attraction with very high demand in hotels and Airbnbs. Therefore, it is important for tourists to observe the characteristics of regions of the city to decide where to stay. 

# **Sources**
# 1. https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm
# 
# 2. https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data?resource=download
# 
# 3. https://data.cccnewyork.org/data/map/66/median-incomes#66/39/2/107/62/a/a
# 
# 4. https://www.criminaljustice.ny.gov/crimnet/ojsa/stats.htm
# 
# 5. https://ny.curbed.com/2019/12/16/20994162/nyc-decade-in-review-rental-market-queens-bronx-corona
# 
# 
