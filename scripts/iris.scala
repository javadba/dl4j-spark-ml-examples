
import sqlContext.implicits._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline

import org.apache.spark.ml.classification.{NeuralNetworkClassification}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration,NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.`override`.{ConfOverride}
import org.deeplearning4j.nn.conf.layers.{RBM,OutputLayer}
import org.nd4j.linalg.lossfunctions.{LossFunctions}

val shellDir = System.getProperty("user.dir")

val iris = MLUtils.loadLibSVMFile(sc, s"file://$shellDir/data/svmLight/iris_svmLight_0.txt").toDF
val irisTrain = iris.sample(false, 0.6, 11L)
val irisTest = iris.except(irisTrain)

val pipeline = (new Pipeline()
  .setStages(Array(
    new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures"),  
    new NeuralNetworkClassification()
    .setFeaturesCol("scaledFeatures")
    .setConf(new NeuralNetConfiguration.Builder()
      .iterations(100)
      .lossFunction(LossFunctions.LossFunction.RMSE_XENT).nIn(4)
      .nOut(3).layer(new RBM())
      .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
      .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
      .activationFunction("tanh").list(2).hiddenLayerSizes(3)
      .`override`(1, new ConfOverride() {
          def overrideLayer(i: Int, builder: NeuralNetConfiguration.Builder) {
              builder.activationFunction("softmax")
              builder.layer(new OutputLayer())
              builder.lossFunction(LossFunctions.LossFunction.MCXENT)
          }
      }).build())
    )))
