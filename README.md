# dl4j-spark-ml-examples

*These instructions are a work in progress, and will be updated soon when binary releases are available.*

## Examples
This repository contains examples of using deeplearning4j with Spark ML.

### Notebooks
A number of examples based on the Spark Notebook:

1. [dl4j-iris](notebooks/dl4j-iris.ipynb) - demonstrates Iris classification using a deep-belief network (Scala)

### Applications
A number of standalone example applications:

1. ml.JavaIrisClassification
2. ml.JavaLfwClassification

## Running
### Notebook

*These instructions are temporary until the next release of Spark Notebook.*

1. Compile and run the Spark Notebook:

```
sbt -D"spark.version"="1.4.0" -D"hadoop.version"="2.2.0" run
```

2. Open the example notebook.   The dl4j-spark-ml package will be automatically loaded.

### Applications
```
$ bin/run-example
Usage: ./bin/run-example <example-class> [example-args]
  - set MASTER=XX to use a specific master
  - can use abbreviated example class name relative to org.deeplearning4j
     (e.g. ml.JavaIrisClassification, ml.JavaLfwClassification)
```
