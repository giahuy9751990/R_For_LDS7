def check_null_nan_pyspark(df):
  from pyspark.sql.functions import col,udf,isnan,when,count
  total_records = df.count()
  for cols in df.columns:
    null_records = df.where(df[cols].isNull()).count()
    nan_records = df.select(isnan('DATE')).toPandas().sum()[0]
    print("Columns " + cols + " has (" + str(null_records) + " nulls AND " + str(nan_records) + " nans) / " + str(total_records) + "(" + str(round((null_records/total_records)*100,2)) + "% percent null)" + " AND (" + str(round((nan_records/total_records)*100,2)) + "% percent nan)")
