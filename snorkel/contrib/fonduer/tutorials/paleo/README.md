# Extracting Formation Measurements from Paleontological Publications

In this advanced tutorial, we will build a `Fonduer` application to tackle the
challenging task of extracting formation measurements from paleontological publications.
This is an example of knowledge base construction from _richly formatted data_.

The entire tutorial can be found in
[`formation_measurement.ipynb`](formation_measurement.ipynb). Before
running the tutorial you will need to:
  1. Run `./download_data.sh` to get the data used in the tutorial.
  2. Create a postgres database named `formation_measurement`. Assuming you have postgres
     installed, you can simply run `createdb formation_measurement`.
