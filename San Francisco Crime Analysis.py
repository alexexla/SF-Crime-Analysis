# Databricks notebook source
# MAGIC %md
# MAGIC ## SF crime data analysis and modeling 

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os
import math
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from ggplot import *
import warnings
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

data_path = "dbfs:/FileStore/tables/Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv"

# COMMAND ----------

# DBTITLE 1,Data preprocessing
crime_data_lines = sc.textFile(data_path)
df_crimes = crime_data_lines.map(lambda line: [x.strip('"') for x in next(reader([line]))])
header = df_crimes.first()
print(header)

#remove the first line of data
crimes = df_crimes.filter(lambda x: x != header)

#get the first line of data
display(df_crimes.take(3))

#get the total number of data 
print("Totol Crime Counts: ", crimes.count())

# COMMAND ----------

# DBTITLE 1,Get dataframe and sql
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_raw = spark.read.format("csv").option("header", "true").load(data_path)
display(df_raw)
df_raw.createOrReplaceTempView("sf_crime")

# COMMAND ----------

# Cleaning raw data
df_raw = df_raw.withColumn('X', df_raw["X"].cast('double'))\
               .withColumn('Y', df_raw["Y"].cast('double'))\
               .withColumn('Day', to_date(col("Date"), "MM/dd/yyyy"))\
               .withColumn('Hour', hour(df_raw["Time"]))

df_clean = df_raw.drop('IncidntNum','Descript','Address','PdId','Time','Date', ':@computed_region_yftq_j783', ':@computed_region_p5aj_wyqh', ':@computed_region_rxqg_mtj9', ':@computed_region_bh8s_q3mv', ':@computed_region_fyvs_ahh9', ':@computed_region_9dfj_4gjx', ':@computed_region_n4xg_c4py', ':@computed_region_4isq_27mq', ':@computed_region_fcz8_est8', ':@computed_region_pigm_ib2e', ':@computed_region_9jxd_iqea', ':@computed_region_6pnf_4xz7', ':@computed_region_6ezc_tdp2', ':@computed_region_h4ep_8xdi', ':@computed_region_nqbw_i6c3', ':@computed_region_2dwj_jsy4')\
                 .withColumnRenamed('Day','Date')

display(df_clean)


# COMMAND ----------

print("Total Number of Crimes from 2003 - 2018: ", df_clean.count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Quick View (OLAP): 

# COMMAND ----------

# DBTITLE 1,Counts the number of crimes for different category.
# Overall
df1 = df_clean.groupBy('category').count().orderBy('count', ascending = False)
df1 = df1.toPandas()
display(df1)

# COMMAND ----------

# Overall 
fig, axes = plt.subplots(1,1, figsize=(15,15))
axes.bar(range(1,25), df1['count'][0:24], align='center')
axes.set_xticks(range(1,25))
axes.set_xticklabels(df1['category'][0:24], rotation=45, fontsize=8)

axes.set_title("Top 25 Category Crimes from 2015-2018")
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC Counts the number of crimes for different district, and visualize results

# COMMAND ----------

fig, axes = plt.subplots(2,3, figsize=(20,20))

df2 = []
year_mark = range(2013,2019)
for i in range(0,6):
  df2.append(df_clean.filter(year(col('Date'))==year_mark[i])\
             .groupBy(df_clean['PdDistrict'].alias('district'))\
             .count().orderBy('count', ascending=False).dropna().toPandas())
  sns.barplot(x='district',y='count',data=df2[i], ax = axes[i//3][i%3])
  axes[i//3][i%3].set_xticklabels(df2[i]['district'],rotation=45,fontsize=10)
  axes[i//3][i%3].set_ylabel('Number of Crimes')
  axes[i//3][i%3].set_title('Number of Crimes in {0}'.format(year_mark[i]))
display(fig)
             

# COMMAND ----------

# Overall
crime_count_by_district = spark.sql("SELECT PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY PdDistrict ORDER BY Count DESC")
display(crime_count_by_district)

# COMMAND ----------

# MAGIC %md
# MAGIC Count the number of crimes each "Sunday" at "SF downtown" from 2013 - 2018

# COMMAND ----------

import matplotlib.dates as mdates
monthsFmt = mdates.DateFormatter('%m')

fig, axes = plt.subplots(2,3, figsize=(12,12))

#fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12,10))
df3 = []
year_mark = range(2013,2019)
for i in range(0,6):
  df3.append(df_clean.filter(df_clean.X > -122.41)\
                     .filter(df_clean.X < -122.40) \
                     .filter(df_clean.Y > 37.79) \
                     .filter(df_clean.Y < 37.80) \
                     .filter(year(col('Date'))==2013) \
                     .filter(col('DayOfWeek')=='Sunday') \
                     .groupBy('Date').count().orderBy('Date').toPandas())
  axes[i//3][i%3].plot(df3[i]['Date'],df3[i]['count'])
  axes[i//3][i%3].xaxis.set_major_formatter(monthsFmt)
  axes[i//3][i%3].set_title('{0}'.format(year_mark[i]))
  axes[i//3][i%3].set_xlabel('month')
  axes[i//3][i%3].set_ylabel('Numbers of Crime on Sunday at SF Downtown')

display(fig)
  

# COMMAND ----------

# MAGIC %md
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018.  

# COMMAND ----------

df4 = df_clean.groupBy(year(col('Date')).alias('year'), month(col('Date')).alias('month')).count().orderBy('year','month').toPandas()
df4_res = df4.pivot(index='month', columns='year', values='count')
ax = df4_res.plot(kind='line', figsize=(15,15))
ax.legend(loc='best')
ax.set_title('Number of crimes in each month from 2003 - 2018', fontsize=20)
display()


# COMMAND ----------

# MAGIC %md
# MAGIC Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15.

# COMMAND ----------

df_15 = df_clean.filter("Date == '2015-02-01'").groupBy('hour').count().orderBy('hour').withColumnRenamed('count','2015').toPandas().set_index('hour')
df_16 = df_clean.filter("Date == '2016-02-01'").groupBy('hour').count().orderBy('hour').withColumnRenamed('count','2016').toPandas().set_index('hour')
df_17 = df_clean.filter("Date == '2017-02-01'").groupBy('hour').count().orderBy('hour').withColumnRenamed('count','2017').toPandas().set_index('hour')

df5 = pd.concat([df_15,df_16,df_17], axis=1)
df5.plot(kind='bar', figsize=(15,10))
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC Find out the top 3 Dangerous disricts and the crime event w.r.t category & time (hour)

# COMMAND ----------

df6 = df_clean.withColumnRenamed('PdDistrict','district').groupBy('district').count().withColumnRenamed('count','2003-2018').orderBy('2003-2018', ascending=False).toPandas().set_index('district')
fig = df6.plot(kind='bar', figsize=(15,15))
xticklabels = list(df6.index)
fig.set_xticklabels(xticklabels, rotation=45, ha='center', fontsize=8)
fig.set_title('Total Crime Count by District -- 2003-2018')
display(fig)

# COMMAND ----------

# MAGIC %md 
# MAGIC Conclusion:
# MAGIC The result shows that the top 3 dangerous district would be: Southern, Mession & Northern

# COMMAND ----------

df6_hour = df_clean.select('*').groupBy('category', 'hour').count().orderBy('hour').toPandas()
df6_hour = df6_hour.pivot(index='category', columns='hour', values='count')
df6_hour = df6_hour.div(df6_hour.sum(axis=0), axis=1)

fig, axs = plt.subplots(6,4, figsize = (25,25))
for i in df6_hour.columns:
  labels = df6_hour[i].sort_values(ascending=False)[:5].index.tolist()
  values = df6_hour[i].sort_values(ascending=False)[:5].values
  labels.append('Other')
  explode = (0.1,0,0,0,0,0)
  values = np.append(values, 1-values.sum())
  axs[i//4][i%4].pie(values, labels=labels, explode=explode, autopct='%1.2f%%', shadow=True,startangle=90)
  axs[i//4][i%4].axis('equal')
  axs[i//4][i%4].set_title('Hour{}'.format(i))
fig.suptitle('Crimes count by Category in 24-Hour scale (2003-2018)')
display()
  

# COMMAND ----------

# MAGIC %md 
# MAGIC For different category of crime, find the percentage of resolution.

# COMMAND ----------

df7_all = df_clean.groupby('Category').count().withColumnRenamed('count', 'all').orderBy('all', ascending=False)
df7_not_null = df_clean.filter(df_clean['Resolution']!='NONE').groupby('Category').count().withColumnRenamed('count', 'resolved').orderBy('resolved',ascending= False)

df7 = df7_all.join(df7_not_null, 'Category').toPandas()
df7['resolution_rate'] = df7['resolved']/df7['all']
df7 = df7.set_index('Category')
df7 = df7.sort_values('resolution_rate', ascending=False)

fig7,ax7 = plt.subplots(figsize = (15,15))
labels = df7.index.tolist()
ax7.barh(np.arange(len(labels)), df7['resolution_rate'], align='center')
ax7.set_yticks(np.arange(len(labels)))
ax7.set_yticklabels(labels, fontsize=5)
ax7.invert_yaxis()
ax7.set_xlabel('Resolution Rate')
ax7.set_title('Resolution rate by category during 2003-2018')
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC visualize the spatial distribution of crimes

# COMMAND ----------

df8 = df_clean.select(['X','Y'])
display(df8)

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

vecAssembler = VectorAssembler(inputCols=["X", "Y"], outputCol="features")
kmeans = KMeans(k=5, seed=1)
pipeline = Pipeline(stages=[vecAssembler, kmeans])
model = pipeline.fit(df8)


df8_res = model.transform(df8)
df8_res = df8_res.toPandas()

# COMMAND ----------

fig8, ax8 = plt.subplots()
ax8.scatter(df8_res['X'], df8_res['Y'], c=(df8_res['prediction']),cmap=plt.cm.jet, alpha=0.9)
ax8.set_xlim([-122.35, -122.55])
ax8.set_ylim([37.70, 37.85])
ax8.set_title("Spacial Distribution")
display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rough Analysis. 
# MAGIC There are several findings can be drawn by going through the dataset. 
# MAGIC According to the above results:
# MAGIC 1. Larceny/Theft would be in the majority of all crimes. 
# MAGIC 2. 'Southern', 'Northern' and 'Mission' are the top three most dangerous districts.
# MAGIC 3. March, August and December presents higher amount of crimes compared to other months, especically on sundays. 
# MAGIC 4. The time range from 6pm to 11pm is on a relatively high crime surge, suggest visitors avoid unnecessary activities at night.
# MAGIC 5. Cases related to traffic, drug and warrants present higher resolution rate than the others.
