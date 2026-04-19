from pyspark.sql.functions import col, expr, when, hour, to_timestamp, lpad, concat_ws, date_add

def union_dataframes(table_path, table_names):
    for table_name in table_names:
        table_full_path = table_path + table_name
        df = spark.table(table_full_path)
        if "jan" in table_name:
            df_Final = df
        else:
            df_Final = df_Final.unionByName(df)
    return df_Final

def create_timestamp(date_col, time_col):
    date_fmt = "M/d/yyyy h:mm:ss a"
    date_only = to_timestamp(col(date_col), date_fmt)
    
    adjusted_time = when(col(time_col) == 2400, 0).otherwise(col(time_col))
    add_day = when(col(time_col) == 2400, 1).otherwise(0)
    
    padded_time = lpad(adjusted_time.cast("int").cast("string"), 4, '0')
    hour = padded_time.substr(1, 2)
    minute = padded_time.substr(3, 2)
    datetime_str = concat_ws(' ', date_add(date_only.cast("date"), add_day), concat_ws(':', hour, minute))
    return datetime_str.cast("timestamp")

from pyspark.sql.functions import col, expr, when, hour, to_timestamp

def prepare_connection_data(df_raw):
    # 1. Timestamps
    df = df_raw.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))
    df = df.withColumn("flight_date", to_timestamp(col("FL_DATE"), "M/d/yyyy h:mm:ss a"))
    df = df.withColumn("schedule_dep", create_timestamp("FL_DATE", "CRS_DEP_TIME"))
    df = df.withColumn("schedule_arr_temp", create_timestamp("FL_DATE", "CRS_ARR_TIME"))
    
    df = df.withColumn(
        "schedule_arr",
        when(col("CRS_ARR_TIME") < col("CRS_DEP_TIME"), 
             expr("from_unixtime(unix_timestamp(schedule_arr_temp) + 86400)").cast("timestamp")
        ).otherwise(col("schedule_arr_temp"))
    ).drop("schedule_arr_temp")

    df = df.withColumn(
        "actual_dept", expr("from_unixtime(unix_timestamp(schedule_dep) + (DEP_DELAY * 60))").cast("timestamp")
    ).withColumn(
        "actual_arr", expr("from_unixtime(unix_timestamp(schedule_arr) + (ARR_DELAY * 60))").cast("timestamp")
    )

    # 2. The Self-Join
    df_A = df.select([col(c).alias(c + "_A") for c in df.columns])
    df_B = df.select([col(c).alias(c + "_B") for c in df.columns])

    connections = df_A.join(
        df_B,
        (col("DEST_A") == col("ORIGIN_B")) & 
        (col("flight_date_A") == col("flight_date_B")) & 
        (expr("unix_timestamp(schedule_dep_B) - unix_timestamp(schedule_arr_A) >= 1800")) & 
        (expr("unix_timestamp(schedule_dep_B) - unix_timestamp(schedule_arr_A) <= 21600"))
    )

    # 3. Feature Engineering
    connections = connections.withColumn(
        "scheduled_layover_mins", expr("(unix_timestamp(schedule_dep_B) - unix_timestamp(schedule_arr_A)) / 60")
    ).withColumn(
        "made_connection", when(expr("unix_timestamp(actual_arr_A) + 1800 <= unix_timestamp(actual_dept_B)"), 1).otherwise(0)
    ).withColumn(
        "is_same_airline", when(col("OP_UNIQUE_CARRIER_A") == col("OP_UNIQUE_CARRIER_B"), 1).otherwise(0)
    ).withColumn(
        "hub_congestion_hour", hour(col("schedule_dep_B"))
    ).withColumn(
        "travel_month", col("flight_date_A").cast("string").substr(6, 2).cast("int")
    )

    return connections.select(
        "OP_UNIQUE_CARRIER_A", "ORIGIN_A", "DEST_A", 
        "is_same_airline", "hub_congestion_hour", "travel_month", "scheduled_layover_mins", "made_connection"
    )

train_flights_path = "workspace.flights.t_ontime_reporting_2024_"
test_flights_path = "workspace.flights.t_ontime_reporting_2025_"
month_list = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

df_Raw2024 = union_dataframes(train_flights_path, month_list)
df_Raw2025 = union_dataframes(test_flights_path, month_list[:3])

df_TrainConnections = prepare_connection_data(df_Raw2024)
df_TestConnections = prepare_connection_data(df_Raw2025)

# STRATIFIED DOWNSAMPLING (Balancing the Train dataset)
# We want roughly ~1 million rows of each class so the model trains fast and fairly.
# 5% of 24.6M misses = ~1.2M rows. 0.2% of 574M successes = ~1.1M rows.
fractions = {0: 0.05, 1: 0.002}
train_balanced = df_TrainConnections.sampleBy("made_connection", fractions, seed=67)

# Materialize the Balanced Training Data (2.4M rows)
# This saves the result of the join to disk so Spark doesn't re-calculate it.
train_balanced.write.mode("overwrite").saveAsTable("workspace.flights.ml_train_gold")

# Materialize a Sample of the Test Data (To speed up evaluation)
# Evaluating on 3 full months of 2025 connections is too much for a free cluster.
# We will take a 10% sample of the 2025 connections to test on (~1-2M rows).
test_sampled = df_TestConnections.sample(False, 0.1, seed=67)
test_sampled.write.mode("overwrite").saveAsTable("workspace.flights.ml_test_sampled")
