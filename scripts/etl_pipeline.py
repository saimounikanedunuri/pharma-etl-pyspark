from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, count, avg, to_date, year, month, dayofmonth,
    upper, lower, lit, udf, desc
)
from pyspark.sql.types import IntegerType, StringType

# Start Spark session
spark = SparkSession.builder.appName("Pharma ETL").getOrCreate()

# Load dataset
df = spark.read.option("header", True).csv("data/pharma_data.csv")

# Initial exploration
df.printSchema()
df.show(5)

# Data Cleaning
df_clean = df.dropna(subset=["Dosage_Milligrams", "Doctor_Name"])
df_clean = df_clean.withColumn("Dosage_Milligrams", col("Dosage_Milligrams").cast("int"))

# Remove duplicates
df_clean = df_clean.dropDuplicates(["Prescription_ID"])

# Business Rule: Flagging
df_enriched = df_clean.withColumn(
    "Dosage_Flag",
    when(col("Dosage_Milligrams") > 500, "High").otherwise("Normal")
)

# Average dosage per medicine
df_dosage_avg = df_enriched.groupBy("Drug_Name").agg(avg("Dosage_Milligrams").alias("Avg_Dosage"))

# Doctor-drug suggestion count
df_doc_suggest = df_enriched.groupBy("Doctor_Name", "Drug_Name")\
    .agg(count("*").alias("Recommended"))\
    .orderBy(desc("Recommended"))

# Extract date components
df_date = df_enriched.withColumn("Prescribed_Date", to_date("Prescribed_Date"))\
    .withColumn("Year", year("Prescribed_Date"))\
    .withColumn("Month", month("Prescribed_Date"))\
    .withColumn("Day", dayofmonth("Prescribed_Date"))

# UDF Example: Risk Level
def get_risk_level(dosage):
    if dosage >= 600:
        return "Critical"
    elif dosage >= 400:
        return "Moderate"
    else:
        return "Low"

risk_udf = udf(get_risk_level, StringType())
df_risk = df_enriched.withColumn("Risk_Level", risk_udf(col("Dosage_Milligrams")))

# Save as Parquet (partitioned)
df_enriched.write.mode("overwrite").partitionBy("Drug_Name").parquet("output/pharma_parquet")

print("✅ ETL pipeline executed successfully.")
