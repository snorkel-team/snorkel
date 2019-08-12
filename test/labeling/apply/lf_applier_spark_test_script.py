"""
This script is used to manually test
    `snorkel.labeling.apply.lf_applier_spark.SparkLFApplier`

To test on AWS EMR:
    1. Allocate an EMR cluster (e.g. label 5.24.0) with > 1 worker and SSH permissions
    2. Clone and pip install snorkel on the master node
        ```
        sudo yum install git
        git clone https://github.com/snorkel-team/snorkel
        cd snorkel
        python3 -m pip install -t snorkel-package .
        cd snorkel-package
        zip -r ../snorkel-package.zip .
        cd ..
        ```
    3. Run
        ```
        sudo sed -i -e \
            '$a\export PYSPARK_PYTHON=/usr/bin/python3' \
            /etc/spark/conf/spark-env.sh
        ```
    4. Run
        ```
        spark-submit \
            --py-files snorkel-package.zip \
            test/labeling/apply/lf_applier_spark_test_script.py
        ```
"""

import logging
from typing import List

import numpy as np
from pyspark import SparkContext

from snorkel.labeling.apply.spark import SparkLFApplier
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
    sc.addPyFile("snorkel-package.zip")
    rdd = sc.parallelize(DATA)

    logging.info("Applying LFs")
    lf_applier = SparkLFApplier([f, g])
    L = lf_applier.apply(rdd)

    np.testing.assert_equal(L.toarray(), L_EXPECTED)


if __name__ == "__main__":
    build_lf_matrix()
