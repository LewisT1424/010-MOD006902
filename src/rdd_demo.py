'''
Comparse the two RDD API alongside the DataFrame API.
'''

import time 
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def status_counts_by_region_rdd(df: DataFrame):
    '''
    RDD: Map each row to ((region, status), 1) then reduceByKey to sum.

    Returns a plain dictionary for easy comparison
    '''
    rdd = df.select('region', 'status_group').rdd
    pair_rdd = rdd.map(lambda row: ((row['region'], row['status_group']), 1))
    counts = pair_rdd.reduceByKey(lambda a, b: a + b)
    return dict(counts.collect())

def status_counts_by_region_dataframe(df: DataFrame) -> DataFrame:
    '''
    DataFrame equivalent (groupBy/Agg)
    '''
    return (
        df.groupBy('region', 'status_group').agg(
            F.count(F.lit(1)).alias('count')
        )
    )

def run_comparison(df: DataFrame) -> dict:
    '''Run both RDD and DataFrame, time them, and sanity-check they agree'''
    df.cache()
    df.count()   # materialise the cache before timing

    t0 = time.time()
    rdd_result = status_counts_by_region_rdd(df) 
    rdd_time = time.time() - t0 

    t0 = time.time()
    df_result = status_counts_by_region_dataframe(df).collect()
    df_time = time.time() - t0
    df_result_dict = {(r['region'], r['status_group']): r['count'] for r in df_result}

    agree = rdd_result == df_result_dict

    return {
        'rdd_time_seconds': round(rdd_time, 4),
        'dataframe_time_seconds': round(df_time, 4),
        'results_match': agree,
        'n_groups': len(rdd_result)
    }