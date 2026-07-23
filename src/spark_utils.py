'''
This module is to detect whether a JVM is discoverable on PATH and if not, installs a local JDK via the 'install-jdk' package and points JAVA_HOME
at it. 
'''

import os 
import shutil

def ensure_java() -> None:
    '''Make sure JVM is discoverable on PATH before Spark tries to launch one'''
    if shutil.which('java') is not None:
        return 


    try:
        import jdk
    except ImportError as e:
        raise RuntimeError(
            "No Java installation found on PATH, and the 'install-jdk' "
            "fallback package is not installed. Run: pip install install-jdk"
        ) from e

    java_home = jdk.install('17')
    os.environ['JAVA_HOME'] = java_home
    os.environ['PATH'] = os.path.join(java_home, 'bin') + os.pathsep + os.environ.get('PATH', '')

def get_spark(app_name: str = 'waterpump', master: str = 'local[*]'):
    '''Return a configured SparkSession, installed Java first if needed'''
    ensure_java()
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config('spark.driver.memory', '2g')
        .getOrCreate()
    )