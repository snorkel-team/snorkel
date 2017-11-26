# Extracting Images for Transistors from PDF Datasheets

In this advanced tutorial, we will build a `Fonduer` application to tackle the
challenging task of extracting images for specific transistor from their 
datasheets. This is an example of knowledge base construction from 
_richly formatted data_.

The entire tutorial can be found in
[`transistor_image_tutorial.ipynb`](transistor_image_tutorial.ipynb). Before
running the tutorial you will need to:
  1. Run `./download_data.sh` to get the data used in the tutorial.
  2. Create a postgres database named `stg_temp_max_figure`. Assuming you have
     postgres installed, you can simply run `createdb stg_temp_max_figure`.
