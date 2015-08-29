/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

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
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassification;
import org.deeplearning4j.spark.sql.sources.iris.DefaultSource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * An iris classification pipeline using a neural network. Derived from
 * {@code org.apache.spark.examples.ml.JavaSimpleTextClassificationPipeline
 * JavaSimpleTextClassificationPipeline}. Run with
 *
 * <pre>
 * bin/run-example ml.JavaIrisClassification
 * </pre>
 */
public class JavaIrisClassification {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("Iris Classification Pipeline (Java)");
        SparkContext jsc = new SparkContext(conf);
        SQLContext jsql = new SQLContext(jsc);

        String path = args.length == 1 ? args[0]
                : "file://" + System.getProperty("user.dir") + "/data/svmLight/iris_svmLight_0.txt";
        DataFrame data = jsql.read()
                .format(DefaultSource.class.getName())
                .load(path);

        System.out.println("\nLoaded IRIS dataframe:");
        data.show(100);

        // prepare train/test set
        DataFrame trainingData = data.sample(false, 0.6, 11L);
        DataFrame testData = data.except(trainingData);

        // Configure an ML pipeline to train a model. In this example,
        // the pipeline combines Spark ML and DL4J elements.
        StandardScaler scaler = new StandardScaler()
                .setWithMean(true).setWithStd(true)
                .setInputCol("features")
                .setOutputCol("scaledFeatures");
        NeuralNetworkClassification classification = new NeuralNetworkClassification()
                .setFeaturesCol("scaledFeatures")
                .setConf(getConfiguration());
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
                scaler, classification });

        // Fit the pipeline on training data.
        System.out.println("\nTraining...");
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions on test data.
        System.out.println("\nTesting...");
        DataFrame predictions = model.transform(testData);

        System.out.println("\nTest Results:");
        predictions.show(100);
    }

    private static MultiLayerConfiguration getConfiguration() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(11L) // Seed to lock in weight initialization for tuning
                .iterations(100) // # training iterations predict/classify & backprop
                .learningRate(1e-3f) // Optimization step size
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT) // Backprop method (calculate the gradients)
                .momentum(0.9)
                .constrainGradientToUnitNorm(true)
                .useDropConnect(true)
                .list(2) // # NN layers (does not count input layer)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                                .nIn(4) // # input nodes
                                .nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
                                .weightInit(WeightInit.XAVIER)
                                .activation("relu")
                                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                                .updater(Updater.ADAGRAD)
                                .k(1) // # contrastive divergence iterations
                                .dropOut(0.5)
                                .build()
                ) // NN layer type
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(3) // # input nodes
                                .nOut(3) // # output nodes
                                .activation("softmax")
                                .weightInit(WeightInit.XAVIER)
                                .updater(Updater.ADAGRAD)
                                .dropOut(0.5)
                                .build()
                ) // NN layer type
                .build();

        return conf;
    }
}
