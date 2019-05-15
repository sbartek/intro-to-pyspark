from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from als_labelizer import Labelizer
from als_model import ALSModel

spark = SparkSession \
    .builder \
    .appName("Pyspark course") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.3")\
    .getOrCreate()

print("Extract data")
order_products_prior_sdf = spark\
    .read.option("header", "true")\
    .csv("data/order_products__prior.csv").cache()

orders_sdf = spark.read.option("header", "true").csv("data/orders.csv")\
    .cache()

print("Get users") 

all_users = orders_sdf.select("user_id").distinct()
all_users, _ = all_users.randomSplit([0.015, 0.985], seed=666)

print("Get items")

all_items = order_products_prior_sdf.select("product_id").distinct()

print("labelize users")

user_labelizer = Labelizer(all_users, "user_id", "user_idx", spark)
user_id_idx = user_labelizer.get_id_idx()
user_labelizer.save("data/user_id_idx.parquet")

print("labelize items")

item_labelizer = Labelizer(all_items, "product_id", "product_idx", spark)
item_id_idx = item_labelizer.get_id_idx()
item_labelizer.save("data/item_id_idx.parquet")

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! create user-item sdf")

user_item_sdf = order_products_prior_sdf.join(orders_sdf, on="order_id")\
    .select(F.col("user_id"), F.col("product_id"), F.lit(1).alias("rating"))\
    .join(user_id_idx, "user_id")\
    .join(item_id_idx, "product_id")\
    .cache()

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! Create model")

als_model = ALSModel("user_idx", "product_idx")
      
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! train model")
    
als_model.fit(user_item_sdf)
als_model.save("data/als_model")

print("DONE !!!!!")
spark.stop()
