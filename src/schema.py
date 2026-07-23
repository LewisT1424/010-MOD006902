'''
Defining schema explicitly (rather than using inferSchema) is a deliberate choice: scheme inference requires an extra full pass over the data to sample types, which does not scale well as data volume grows,
and it is fragile against malformed rows silently being cast to string.
'''

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType, BooleanType

TRAIN_VALUES_SCHEMA = StructType([
    StructField('id', LongType(), False),
    StructField('amount_tsh', DoubleType(), True),
    StructField('date_recorded', StringType(), True), # Parsed to DateType later
    StructField('funder', StringType(), True),
    StructField('gps_height', IntegerType(), True),
    StructField('installer', StringType(), True),
    StructField('longitude', DoubleType(), True),
    StructField('latitude', DoubleType(), True),
    StructField('wpt_name', StringType(), True),
    StructField('num_private', IntegerType(), True),
    StructField("basin", StringType(), True),
    StructField("subvillage", StringType(), True),
    StructField("region", StringType(), True),
    StructField("region_code", IntegerType(), True),
    StructField("district_code", IntegerType(), True),
    StructField("lga", StringType(), True),
    StructField("ward", StringType(), True),
    StructField("population", IntegerType(), True),
    StructField("public_meeting", BooleanType(), True),
    StructField("recorded_by", StringType(), True),
    StructField("scheme_management", StringType(), True),
    StructField("scheme_name", StringType(), True),
    StructField("permit", BooleanType(), True),
    StructField("construction_year", IntegerType(), True),
    StructField("extraction_type", StringType(), True),
    StructField("extraction_type_group", StringType(), True),
    StructField("extraction_type_class", StringType(), True),
    StructField("management", StringType(), True),
    StructField("management_group", StringType(), True),
    StructField("payment", StringType(), True),
    StructField("payment_type", StringType(), True),
    StructField("water_quality", StringType(), True),
    StructField("quality_group", StringType(), True),
    StructField("quantity", StringType(), True),
    StructField("quantity_group", StringType(), True),
    StructField("source", StringType(), True),
    StructField("source_type", StringType(), True),
    StructField("source_class", StringType(), True),
    StructField("waterpoint_type", StringType(), True),
    StructField("waterpoint_type_group", StringType(), True),

])

TRAIN_LABELS_SCHEMA = StructType([
    StructField('id', LongType(), False),
    StructField('status_group', StringType(), False),
])