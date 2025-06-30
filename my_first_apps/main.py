import os
import shutil
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
import pyspark.sql.functions as funcs
from pyspark import SparkConf

# === Configuration ===
TEMP_DIR = "/opt/bitnami/spark/spark-files/files/temp"
PROCESSED_DIR = "/opt/bitnami/spark/spark-files/files/processed"
DONE_DIR = "/opt/bitnami/spark/spark-files/files/done"
MAX_FILES = 5

def main():
    # Create directories if they don't exist
    for directory in [TEMP_DIR, PROCESSED_DIR, DONE_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # === Spark session ===
    spark = (
        SparkSession.builder
        .appName("clickhouse-batch-ingest-3")
        .config("spark.driver.port", "4040")
        .config("spark.blockManager.port", "4041")
        .master("spark://spark-master:7077")  # Use container name instead of localhost
        .config("spark.driver.host", "pyspark-app")
        .config("spark.driver.port", "4040")
        .config("spark.blockManager.port", "4041")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.jars.ivy", "/tmp/.ivy2")
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
        spark.stop()
        return

    print(f"Found {len(selected_files)} files to process:")
    for f in selected_files:
        print(f"  - {f.name}")

    # === Move selected files to 'processed/' ===
    for f in selected_files:
        destination = os.path.join(PROCESSED_DIR, f.name)
        print(f"Moving {f} to {destination}")
        shutil.move(str(f), destination)

    # === Build paths from processed directory ===
    processed_paths = [str(Path(PROCESSED_DIR) / f.name) for f in selected_files]
    print(f"Processing files: {processed_paths}")

    # === Read the selected files ===
    try:
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
        null_dates = spark.sql("SELECT * FROM transactions WHERE order_date IS NULL")
        print(f"Records with null order_date: {null_dates.count()}")
        if null_dates.count() > 0:
            null_dates.show(5)

        print(f"Total records to process: {output.count()}")

        # === Write to ClickHouse ===
        print("Writing to ClickHouse...")
        (
            output.write
            .format("jdbc")
            .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
            .option("url", "jdbc:clickhouse://192.168.1.9:8123/default")
            .option("dbtable", "sales")
            .option("user", "wills")
            .option("password", "wills")
            .mode("append")
            .save()
        )
        print("Successfully wrote to ClickHouse!")

        # === Move processed files to 'done/' only if write succeeded ===
        for f in selected_files:
            processed_path = os.path.join(PROCESSED_DIR, f.name)
            done_path = os.path.join(DONE_DIR, f.name)
            print(f"Moving {processed_path} to {done_path}")
            shutil.move(processed_path, done_path)

        print(f"Successfully processed {len(selected_files)} file(s).")

    except Exception as e:
        print(f"Error processing files: {str(e)}")
        # Move files back to temp directory if processing failed
        for f in selected_files:
            processed_path = os.path.join(PROCESSED_DIR, f.name)
            if os.path.exists(processed_path):
                shutil.move(processed_path, str(f))
                print(f"Moved {f.name} back to temp directory due to error")
        raise e

    finally:
        spark.stop()

if __name__ == "__main__":
    main()