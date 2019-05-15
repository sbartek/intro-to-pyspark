from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from als_labelizer import Labelizer
from als_model import ALSRecModel

spark = SparkSession \
    .builder \
    .appName("Pyspark course") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.3")\
    .getOrCreate()

print("read user and item id to idx")
user_labelizer = Labelizer(None, "user_id", "user_idx", spark)
user_labelizer.load("data/user_id_idx.parquet")
user_id_idx = user_labelizer.id_idx

item_labelizer = Labelizer(None, "product_id", "product_idx", spark)
item_labelizer.load("data/item_id_idx.parquet")
item_id_idx = item_labelizer.id_idx

print("read model")

als_model = ALSRecModel("user_idx", "product_idx", spark=spark)
als_model.load("data/als_model")

print("predict")

prediction = als_model.transform()\
    .select("user_idx", F.explode("recommendations").alias("rec"))\
    .select("user_idx", "rec.product_idx")

prediction.show()
prediction\
    .repartition(1)\
    .write.mode("overwrite")\
    .option("header","true")\
    .csv("data/prediction/")
prediction.show()

spark.stop()
