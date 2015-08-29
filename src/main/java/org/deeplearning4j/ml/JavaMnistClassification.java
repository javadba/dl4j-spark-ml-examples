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
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
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

        DataFrame trainingData = data.sample(false, 0.8, 123);
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

        final int numRows = 28;
        final int numColumns = 28;
        int nChannels = 1;
        int outputNum = 10;
        int numSamples = 2000;
        int batchSize = 500;
        int iterations = 10;
        int seed = 123;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(nChannels)
                        .nOut(6)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(150)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(numRows, numColumns, 1))
                .inputPreProcessor(2, new CnnToFeedForwardPreProcessor())
                .backprop(true).pretrain(false)
                .build();

        return conf;
    }
}
