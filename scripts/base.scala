
import sqlContext.implicits._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline

import org.deeplearning4j.spark.ml.classification.{NeuralNetworkClassification}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration,NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.`override`.{ConfOverride}
import org.deeplearning4j.nn.conf.layers.{RBM,OutputLayer}
import org.nd4j.linalg.lossfunctions.{LossFunctions}

