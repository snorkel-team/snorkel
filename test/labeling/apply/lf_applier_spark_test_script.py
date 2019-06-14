"""
This script is used to manually test
    `snorkel.labeling.apply.lf_applier_spark.SparkLFApplier`

To test on AWS EMR:
    1. Allocate an EMR cluster (e.g. label 5.20.0) with > 1 worker and
        `PYSPARK_PYTHON` set to `/usr/bin/python3` in the `yarn-env`
    2. Clone and pip install snorkel on the master node
    3. Run
        ```
        spark-submit \
            --py-files path/to/snorkel-x.y.zip \
            lf_applier_spark_test_script.py
        ```
"""

import logging
from typing import List

import numpy as np
from pyspark import SparkContext

from snorkel.labeling.apply.lf_applier_spark import SparkLFApplier
from snorkel.labeling.lf import labeling_function
from snorkel.types import DataPoint

logging.basicConfig(level=logging.INFO)


@labeling_function()
def f(x: DataPoint) -> int:
    return 1 if x.a > 42 else 0


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return 1 if x.a in db else 0


DATA = [3, 43, 12, 9]
L_EXPECTED = np.array([[0, 1], [1, 0], [0, 0], [0, 1]])


def build_lf_matrix() -> None:

    logging.info("Getting Spark context")
    sc = SparkContext()
    rdd = sc.parallelize(DATA)

    logging.info("Applying LFs")
    lf_applier = SparkLFApplier([f, g])
    L = lf_applier.apply(rdd)

    np.testing.assert_equal(L.toarray(), L_EXPECTED)


if __name__ == "__main__":
    build_lf_matrix()
