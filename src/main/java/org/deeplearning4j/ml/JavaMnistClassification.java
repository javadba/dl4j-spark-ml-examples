package org.deeplearning4j.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassification;
import org.deeplearning4j.spark.sql.sources.mnist.DefaultSource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * An Mnist classification pipeline using a neural network. Derived from
 * {@code org.apache.spark.examples.ml.JavaSimpleTextClassificationPipeline}
 *
 * Run with
 * <pre>
 * bin/run-example ml.JavaMnistClassification
 * </pre>
 */
public class JavaMnistClassification {

    final static int numRows = 28;
    final static int numColumns = 28;
    final static int outputNum = 10;
    final static int iterations = 10;
    final static int seed = 123;

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("Mnist Classification Pipeline (Java)");
        SparkContext jsc = new SparkContext(conf);
        SQLContext jsql = new SQLContext(jsc);

        String imagesPath = args.length == 2 ? args[0]
                : "file://" + System.getProperty("user.dir") + "/data/train-images-idx3-ubyte";
        String labelsPath = args.length == 2 ? args[1]
                : "file://" + System.getProperty("user.dir") + "/data/train-labels-idx1-ubyte";
        Map<String, String> params = new HashMap<String, String>();
        params.put("imagesPath", imagesPath);
        params.put("labelsPath", labelsPath);
        DataFrame data = jsql.read().format(DefaultSource.class.getName())
                .options(params).load();

        System.out.println("\nLoaded Mnist dataframe:");
        data.show(100);

        DataFrame trainingData = data.sample(false, 0.6, 11L);
        DataFrame testData = data.except(trainingData);

        StandardScaler scaler = new StandardScaler()
                .setWithMean(true).setWithStd(true)
                .setInputCol("features")
                .setOutputCol("scaledFeatures");
        NeuralNetworkClassification classification = new NeuralNetworkClassification()
                .setFeaturesCol("scaledFeatures")
                .setConf(getConfiguration());
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                scaler, classification});

        System.out.println("\nTraining...");
        PipelineModel model = pipeline.fit(trainingData);

        System.out.println("\nTesting...");
        DataFrame predictions = model.transform(testData);

        System.out.println("\nTest Results:");
        predictions.show(100);
    }

    public static MultiLayerConfiguration getConfiguration() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .iterations(iterations).l1(1e-1).l2(1e-3).constrainGradientToUnitNorm(true)
                .regularization(true)
                .visibleUnit(RBM.VisibleUnit.BINARY)
                .hiddenUnit(RBM.HiddenUnit.BINARY)
                .list(4)
                .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(600).build())
                .layer(1, new RBM.Builder().nIn(600).nOut(400).build())
                .layer(2, new RBM.Builder().nIn(400).nOut(200).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")
                        .nIn(200).nOut(outputNum).build())
                .pretrain(true).backprop(false)
                .build();

        return conf;
    }
}
