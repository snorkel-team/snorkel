# Extracting Maximum Storage Temperatures from Transistor Datasheets

In this advanced tutorial, we will build a `Fonduer` application to tackle the
challenging task of extracting maximum storage temperatures for specific
transistor part numbers from their datasheets. This is an example of knowledge
base construction from _richly formatted data_.

The entire tutorial can be found in
[`max_storage_temp_tutorial.ipynb`](max_storage_temp_tutorial.ipynb). Before
running the tutorial you will need to:
  1. Run `./download_data.sh` to get the data used in the tutorial.
  2. Create a postgres database named `stg_temp_max`. Assuming you have postgres
     installed, you can simply run `createdb stg_temp_max`.


## Example

For example, the simplified datasheet snippet:

![datasheet-snippet](../img/sample-datasheet.png)

our goal is to extract the (transistor part number, maximum storage temperature)
 relation pairs:

```
("SMBT3904", "150")
("MMBT3904", "150")
```
