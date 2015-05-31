package org.deeplearning4j.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassification;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.spark.sql.sources.lfw.*;
/**
 * An LFW classification pipeline using a neural network. Derived from
 * {@code org.apache.spark.examples.ml.JavaSimpleTextClassificationPipeline
 * JavaSimpleTextClassificationPipeline}. Run with
 *
 * <pre>
 * bin/run-example ml.JavaLfwClassification
 * </pre>
 */
public class JavaLfwClassification {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("LFW Classification (Java)");
        SparkContext jsc = new SparkContext(conf);
        SQLContext jsql = new SQLContext(jsc);

        if(args.length != 1) {
            System.out.println("usage: run-example ml.JavaLfwClassification <URI>");
            System.out.println("where:\n\tURI: filesystem path to the lFW dataset");
            return;
        }
        String path = args[0];
        DataFrame data = new JavaLfwContext(jsql).lfw(path);

        System.out.println("\nLoaded LFW dataframe:");
        data.show(100);

        // prepare train/test set
        DataFrame trainingData = data.sample(false, 0.6, 11L);
        DataFrame testData = data.except(trainingData);

        // Configure an ML pipeline to train a model. In this example,
        // the pipeline combines Spark ML and DL4J elements.
        StringIndexer indexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("labelIndex");
        StandardScaler scaler = new StandardScaler()
                .setWithMean(true).setWithStd(true)
                .setInputCol("features").setOutputCol("scaledFeatures");
        NeuralNetworkClassification classification = new NeuralNetworkClassification()
                .setLabelCol("labelIndex")
                .setFeaturesCol("scaledFeatures")
                .setConf(getConfiguration(data));
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
                indexer, scaler, classification });

        // Fit the pipeline on training data.
        System.out.println("\nTraining...");
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions on test data.
        System.out.println("\nTesting...");
        DataFrame predictions = model.transform(testData);

        System.out.println("\nTest Results:");
        predictions.show(100);
    }

    private static MultiLayerConfiguration getConfiguration(DataFrame dataset) {

        //int numLabels = (int) dataset.select("label").count();
        int numLabels = 20000;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(28 * 28)
                .nOut(numLabels)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(1e-3, 1e-1))
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .constrainGradientToUnitNorm(true)
                .learningRate(1e-3f)
                .list(4)
                .hiddenLayerSizes(600, 250, 200)
                .override(3, new ClassifierOverride())
                .build();

        return conf;
    }
}
