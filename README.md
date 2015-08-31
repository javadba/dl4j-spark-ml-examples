# dl4j-spark-ml-examples

## Examples
This repository contains examples of using deeplearning4j with Spark ML.

### Notebooks
A number of examples based on the Spark Notebook:

1. [dl4j-iris](notebooks/dl4j-iris.ipynb) - demonstrates Iris classification using a deep-belief network (Scala)

### Applications
A number of standalone example applications:

1. ml.JavaIrisClassification
2. ml.JavaLfwClassification
3. ml.JavaMnistClassification (* broken in dl4j rc0)

## Compile
1. Compile project with maven.

```
$ mvn clean package -Dspark.version=1.4.0 -Dhadoop.version=2.2.0
```

## Running

### Notebook

*These instructions are temporary until the next release of Spark Notebook.*
1
. Open the example notebook.   The dl4j-spark-ml package will be automatically loaded.

### Applications
Before running example application, it is necessary to set up `SPARK_HOME` env variable.

```
$ export SPARK_HOME=<Your Spark Path>
$ bin/run-example
Usage: ./bin/run-example <example-class> [example-args]
  - set MASTER=XX to use a specific master
  - can use abbreviated example class name relative to org.deeplearning4j
     (e.g. ml.JavaIrisClassification, ml.JavaLfwClassification)
```

For example,

```
$ bin/run-example ml.JavaIrisClassification
```
