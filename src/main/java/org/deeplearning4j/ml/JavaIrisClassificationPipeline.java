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
import org.apache.spark.ml.classification.NeuralNetworkClassification;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * An iris classification pipeline using a neural network. Derived from
 * {@link org.apache.spark.examples.ml.JavaSimpleTextClassificationPipeline
 * JavaSimpleTextClassificationPipeline}. Run with
 * 
 * <pre>
 * bin/run-example ml.JavaIrisClassificationPipeline
 * </pre>
 */
public class JavaIrisClassificationPipeline {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("Iris Classification Pipeline (Java)");
        SparkContext jsc = new SparkContext(conf);
        SQLContext jsql = new SQLContext(jsc);

        String path = args.length == 1 ? args[0]
                : "data/svmLight/iris_svmLight_0.txt";
        DataFrame data = jsql.createDataFrame(
                MLUtils.loadLibSVMFile(jsc, path), LabeledPoint.class);

        System.out.println("\nLoaded IRIS dataframe:");
        data.show(100);

        // prepare train/test set
        DataFrame trainingData = data.sample(false, 0.6, 11L);
        DataFrame testData = data.except(trainingData);

        // Configure an ML pipeline to train a model. In this example,
        // the pipeline combines Spark ML and DL4J elements.
        StandardScaler scaler = new StandardScaler()
                // .setWithMean(true).setWithStd(true) /* Spark 1.4 */
                .setInputCol("features").setOutputCol("scaledFeatures");
        NeuralNetworkClassification classification = new NeuralNetworkClassification()
                .setFeaturesCol("scaledFeatures").setConf(getConfiguration());
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
                .iterations(100)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT).nIn(4)
                .nOut(3).layer(new RBM())
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .activationFunction("tanh").list(2).hiddenLayerSizes(3)
                .override(1, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i,
                            NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new OutputLayer());
                        builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                    }
                }).build();

        return conf;
    }
}
