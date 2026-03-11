import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def main(input_path: str, output_path: str) -> None:
    spark = SparkSession.builder \
        .appName("Boston Crimes Vitrine") \
        .getOrCreate()
    crimes = spark.read.csv(
        f"{input_path}/crime.csv",
        header=True,
        inferSchema=True,
    )
    offense_codes = spark.read.csv(
        f"{input_path}/offense_codes.csv",
        header=True,
        inferSchema=True,
    )
    crimes = crimes.dropDuplicates()
    offense_codes = offense_codes.dropDuplicates(["CODE"])
    offense_codes = offense_codes.withColumn(
        "crime_type",
        F.trim(F.split(F.col("NAME"), " - ").getItem(0)),
    )

    crimes = crimes.join(
        F.broadcast(offense_codes.select("CODE", "crime_type")),
        crimes["OFFENSE_CODE"] == offense_codes["CODE"],
        "left",
    ).drop("CODE")
    crimes_total = crimes.groupBy("DISTRICT") \
        .agg(F.count("*").alias("crimes_total"))

    monthly_counts = crimes.groupBy("DISTRICT", "YEAR", "MONTH") \
        .agg(F.count("*").alias("month_cnt"))

    crimes_monthly = monthly_counts.groupBy("DISTRICT") \
        .agg(
            F.expr("percentile_approx(month_cnt, 0.5)").alias("crimes_monthly"),
        )

    # 4c. frequent_crime_types — top-3 crime types per district
    crime_type_counts = crimes.groupBy("DISTRICT", "crime_type") \
        .agg(F.count("*").alias("ct_count"))

    w = Window.partitionBy("DISTRICT").orderBy(F.col("ct_count").desc())

    top3 = crime_type_counts \
        .withColumn("rn", F.row_number().over(w)) \
        .filter(F.col("rn") <= 3) \
        .groupBy("DISTRICT") \
        .agg(
            F.concat_ws(
                ", ",
                F.collect_list(F.col("crime_type")),
            ).alias("frequent_crime_types"),
        )
    coords = crimes.groupBy("DISTRICT") \
        .agg(
            F.avg("Lat").alias("lat"),
            F.avg("Long").alias("lng"),
        )
    vitrine = crimes_total \
        .join(crimes_monthly, "DISTRICT") \
        .join(top3, "DISTRICT") \
        .join(coords, "DISTRICT")
    vitrine = vitrine.filter(
        F.col("DISTRICT").isNotNull() & (F.col("DISTRICT") != "")
    )
    vitrine.coalesce(1).write.mode("overwrite").parquet(output_path)

    print("Vitrine saved to", output_path)
    vitrine.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit boston_crimes.py <input_path> <output_path>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
