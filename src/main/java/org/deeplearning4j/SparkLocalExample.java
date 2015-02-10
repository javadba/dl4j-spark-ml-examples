package org.deeplearning4j;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.BinomialSamplingPreProcessor;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.deeplearning4j.spark.canova.RDDMiniBatches;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by agibsonccc on 1/22/15.
 */
public class SparkLocalExample {

    public static void main(String[] args) throws Exception {

        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[8]").set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION,"false")
                .set("spark.akka.frameSize", "100")
                .setAppName("mnist");

        System.out.println("Setting up Spark Context...");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        Map<Integer,OutputPreProcessor> preProcessorMap = new HashMap<>();
        for(int i = 0; i < 3; i++)
            preProcessorMap.put(i,new BinomialSamplingPreProcessor());

        int batchSize = 5000;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(0.9).iterations(1)
                .constrainGradientToUnitNorm(true).weightInit(WeightInit.DISTRIBUTION)
                .dist(Distributions.normal(new MersenneTwister(123), 1e-4))
                .nIn(784).nOut(10).layerFactory(LayerFactories.getFactory(RBM.class))
                .list(4).hiddenLayerSizes(600, 500, 400)
                .override(new NeuralNetConfiguration.ConfOverride() {
                    @Override
                    public void override(int i, NeuralNetConfiguration.Builder builder) {

                        if (i == 3) {
                            builder.activationFunction(Activations.softMaxRows());
                            builder.layerFactory(LayerFactories.getFactory(OutputLayer.class));
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        }
                    }
                }).build();




        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);
        DataSet d = new MnistDataSetIterator(60000,60000).next();
        List<DataSet> next = new ArrayList<>();
        for(int i = 0; i < d.numExamples(); i++)
            next.add(d.get(i).copy());

        JavaRDD<DataSet> data = sc.parallelize(next);
        MultiLayerNetwork network2 = master.fitDataSet(data);

        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(),network2.output(d.getFeatureMatrix()));
        System.out.println("Averaged once " + evaluation.stats());


        INDArray params = network2.params();
        Nd4j.writeTxt(params,"params.txt",",");
        FileUtils.writeStringToFile(new File("conf.json"),network2.getLayerWiseConfigurations().toJson());
    }

}
