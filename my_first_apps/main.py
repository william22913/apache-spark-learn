import os
import shutil
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
import pyspark.sql.functions as funcs

# === Configuration ===
TEMP_DIR = "./files/temp"
PROCESSED_DIR = "./files/processed"
DONE_DIR = "./files/done"
MAX_FILES = 5

# === Spark session ===
spark = (
    SparkSession.builder
    .appName("clickhouse-batch-ingest")
    .config("spark.jars", "clickhouse-jdbc-0.6.0-all.jar")
    .getOrCreate()
)

# === CSV schema ===
schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("region", StringType(), True),
    StructField("age", DoubleType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("total_price", DoubleType(), True),
    StructField("shipping_fee", DoubleType(), True),
    StructField("shipping_status", StringType(), True),
    StructField("order_date", DateType(), True)
])

# === Select up to 5 CSV files sorted by creation time ===
all_files = sorted(Path(TEMP_DIR).glob("*.csv"), key=lambda f: f.stat().st_ctime)
selected_files = all_files[:MAX_FILES]

if not selected_files:
    print("No new files to process.")
    exit(0)

# === Move selected files to 'processed/' ===
for f in selected_files:
    shutil.move(str(f), os.path.join(PROCESSED_DIR, f.name))

# === Build paths from processed directory ===
processed_paths = [str(Path(PROCESSED_DIR) / f.name) for f in selected_files]

# === Read the selected files ===
transaction_data = (
    spark.read.format("csv")
    .schema(schema)
    .option("nullValue", "")
    .option("header", "true")
    .option("treatEmptyValuesAsNulls", "true")
    .load(processed_paths)
)

# === Transformations ===
output = (
    transaction_data
    .withColumn("insert_at", funcs.current_timestamp())
    .withColumn("source_file", funcs.regexp_extract(funcs.input_file_name(), r"([^/\\]+$)", 1))  # Extract file name once
    .withColumn("metadata", funcs.to_json(funcs.struct(
        funcs.col("source_file").alias("source_file"),
        funcs.lit("v2").alias("pipeline")
    )))
    .drop("source_file")  # Clean up temp column
    .orderBy("order_date")
)


# === Create temp view for debugging (optional) ===
output.createOrReplaceTempView("transactions")
spark.sql("SELECT * FROM transactions WHERE order_date IS NULL").show()

# === Write to ClickHouse ===
(
    output.write
    .format("jdbc")
    .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
    .option("url", "jdbc:clickhouse://localhost:8123/default")
    .option("dbtable", "sales")
    .option("user", "wills")
    .option("password", "wills")
    .mode("append")
    .save()
)

# === Move processed files to 'done/' only if write succeeded ===
for f in selected_files:
    processed_path = os.path.join(PROCESSED_DIR, f.name)
    shutil.move(processed_path, os.path.join(DONE_DIR, f.name))

print(f"Successfully processed {len(selected_files)} file(s).")