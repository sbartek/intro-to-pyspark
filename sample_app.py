from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

df = spark.createDataFrame(
    [(0, 0, 4.0), (0, 1, 2.0), (0, 3, 3.0), (1, 0, 4.0), (1, 1, 1.0), (1, 2, 5.0)],
    ["user", "item", "rating"]
)

df_pandas = df.groupBy("user").agg(F.count(F.col("item"))).toPandas()
print(df_pandas)

spark.stop()
