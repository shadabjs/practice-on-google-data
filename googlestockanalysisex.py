from pyspark.sql import SparkSession
from pyspark.sql.functions import max, min, mean, corr

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("GoogleStockAnalysis") \
    .getOrCreate()

# Load the CSV file into a DataFrame
df = spark.read.csv("GOOGL_data.csv", header=True, inferSchema=True)

# Display the schema and first few rows of the DataFrame
df.printSchema()
df.show(5)

# Q.1.What is the highest closing price recorded for Google stock?
max_closing_price = df.agg(max("close")).collect()[0][0]
print("Highest closing price:", max_closing_price)

# Q.2.On which date did Google stock have the lowest volume traded?
min_volume_date = df.orderBy("volume").select("date").first()[0]
print("Date with lowest volume traded:", min_volume_date)

# Q.3.What was the difference between the highest and lowest prices for Google stock on a specific date?
specific_date = '2013-02-15'  # Replace with the desired date
price_range = df.filter(df.date == specific_date).select((max("high") - min("low")).alias("price_range")).first()[0]
print("Price range on", specific_date, ":", price_range)

# Q.4.How many days did Google stock close higher than its opening price?
higher_closing_days = df.filter(df.close > df.open)
num_higher_closing_days = higher_closing_days.count()
print("Number of days Google stock closed higher than opening price:", num_higher_closing_days)

# Q.5.What was the average volume of Google stock traded over the entire dataset?
average_volume = df.agg(mean("volume")).collect()[0][0]
print("Average volume traded:", average_volume)

# Q.6.Can you identify any outliers in the volume traded for Google stock? If so, on which dates did they occur?
# Let's define outliers as volume greater than 3 standard deviations from the mean
std_dev = df.agg({"volume": "stddev"}).collect()[0][0]
mean_volume = df.agg({"volume": "mean"}).collect()[0][0]
outliers = df.filter(df.volume > mean_volume + 3 * std_dev)
print("Outliers in volume traded:")
outliers.select("date", "volume").show()

# Q.7.is there any correlation between the opening and closing prices of Google stock?
correlation = df.stat.corr("open", "close")
print("Correlation between opening and closing prices:", correlation)

# Q.8.How many times did Google stock hit its highest price during the given period?
max_price_hits = df.filter(df.high == df.agg(max("high")).collect()[0][0])
num_max_price_hits = max_price_hits.count()
print("Number of times Google stock hit its highest price:", num_max_price_hits)


# Q.9.Grouping by month and calculating average closing price:
from pyspark.sql.functions import year, month
monthly_avg_closing = df.groupBy(year("date").alias("year"), month("date").alias("month")) \
    .agg({"close": "avg"}) \
    .orderBy("year", "month")
# Show the result
monthly_avg_closing.show()


# Q.10.Calculating the total traded volume for each month:
from pyspark.sql.functions import year, month, sum
# Group by year and month, calculate total traded volume
monthly_total_volume = df.groupBy(year("date").alias("year"), month("date").alias("month")) \
    .agg({"volume": "sum"}) \
    .orderBy("year", "month")
# Show the result
monthly_total_volume.show()



# Q.11.Finding the date with the highest closing price:
from pyspark.sql.functions import max
# Find the date with the highest closing price
max_close_date = df.filter(df['close'] == df.agg(max("close")).collect()[0][0]) \
    .select("date") \
    .collect()[0][0]
print("Date with the highest closing price:", max_close_date)



# Q.12.Filtering data for weekdays only:
from pyspark.sql.functions import dayofweek
# Filter data for weekdays only
weekdays_data = df.filter(dayofweek("date").isin([2, 3, 4, 5, 6]))  # Assuming Monday=2, Tuesday=3, ..., Friday=6
# Show the result
weekdays_data.show()




# Q.13.Calculating the average closing price per year:
from pyspark.sql.functions import year, avg
# Group by year, calculate average closing price
yearly_avg_closing = df.groupBy(year("date").alias("year")) \
    .agg({"close": "avg"}) \
    .orderBy("year")
# Show the result
yearly_avg_closing.show()



# Q.14.Finding the date with the largest price change:
from pyspark.sql.functions import abs, greatest
# Add a column for absolute price change
df = df.withColumn("abs_price_change", abs(df['close'] - df['open']))
# Find the date with the largest price change
max_price_change_date = df.filter(df['abs_price_change'] == df.agg(max("abs_price_change")).collect()[0][0]) \
    .select("date") \
    .collect()[0][0]
print("Date with the largest price change:", max_price_change_date)



# Q.15.Identifying the top N dates with the highest trading volume:
from pyspark.sql.functions import desc
N = 5  # Number of top dates to identify
# Get the top N dates with the highest trading volume
top_dates = df.orderBy(desc("volume")).limit(N).select("date")
# Show the result
top_dates.show()


# Q.16.Computing the average closing price for each quarter:
from pyspark.sql.functions import quarter
# Group by year and quarter, calculate average closing price
quarterly_avg_closing = df.groupBy(year("date").alias("year"), quarter("date").alias("quarter")) \
    .agg({"close": "avg"}) \
    .orderBy("year", "quarter")
# Show the result
quarterly_avg_closing.show()


# Q.17.Finding the date with the highest intra-day price range (difference between high and low):
# Add a column for intra-day price range
df = df.withColumn("intraday_range", df["high"] - df["low"])
# Find the date with the highest intra-day price range
max_range_date = df.filter(df["intraday_range"] == df.agg(max("intraday_range")).collect()[0][0]) \
    .select("date") \
    .collect()[0][0]
print("Date with the highest intra-day price range:", max_range_date)



# Q.18.Calculating the average trading volume per year:
from pyspark.sql.functions import year, avg
# Group by year, calculate average trading volume
avg_volume_per_year = df.groupBy(year("date").alias("year")) \
                        .agg(avg("volume").alias("avg_volume")) \
                        .orderBy("year")
# Show the result
avg_volume_per_year.show()


# Stop the Spark session
spark.stop()
