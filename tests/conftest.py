"""
Shared pytest fixtures. A single SparkSession is reused
across all tests rather than created per-test
"""

import pytest

from src.spark_utils import get_spark


@pytest.fixture(scope="session")
def spark():
    spark_session = get_spark(app_name="pytest-waterpump", master="local[1]")
    spark_session.sparkContext.setLogLevel("ERROR")
    yield spark_session
    spark_session.stop()