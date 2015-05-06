package org.deeplearning4j;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.preprocessor.BinomialSamplingPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

/**
 * Created by agibsonccc on 1/22/15.
 */
public class DistributedExample {
    @Option(name="--batchSize")
    private int batchSize = 10000;
    @Option(name = "--frameSize")
    private int frameSize = 600000;
    @Option(name = "--masterUrl",required = true)
    private String masterUrl;
    @Option(name = "--iterations")
    private int iterations = 5;
    @Option(name = "--output")
    private String outputPath = "mnist.ser";
    @Option(name = "--avgiteration")
    private boolean averageEachIteration = false;





    private static Logger log = LoggerFactory.getLogger(DistributedExample.class);

    public DistributedExample(String[] args) {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        }
        catch (CmdLineException e) {
            parser.printUsage(System.err);
            log.error("Unable to parse args",e);
        }
    }

    public static void main(String[] args) throws Exception {

        DistributedExample app = new DistributedExample(args);
        // set to test mode
        SparkConf sparkConf = new SparkConf().set("spark.executor.extraJavaOptions","-Ddtype=float")
                .setMaster(app.masterUrl).set("spark.akka.frameSize", String.valueOf(app.frameSize))
                .set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(app.averageEachIteration))
                .setAppName("mnist");

        System.out.println("Setting up Spark Context...");
        Nd4j.dtype = DataBuffer.Type.FLOAT;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        INDArray n = Nd4j.create(5);

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        Map<Integer,OutputPreProcessor> preProcessorMap = new HashMap<>();
        preProcessorMap.put(0,new BinomialSamplingPreProcessor());
        preProcessorMap.put(1,new BinomialSamplingPreProcessor());
        preProcessorMap.put(2,new BinomialSamplingPreProcessor());

        MultiLayerConfiguration conf = new NeuralNetConfiguration
                .Builder().iterations(app.iterations).momentum(0.5)
                .l2(2e-4).regularization(true)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .nIn(784).nOut(10)
                .layer(new RBM()).batchSize(app.batchSize).momentumAfter(Collections.singletonMap(20, 0.9))
                .list(4).hiddenLayerSizes(600, 500, 400)
                .override(3, new ClassifierOverride(3)).build();



        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);
        DataSet d = new MnistDataSetIterator(60000,60000).next();
        List<DataSet> next = new ArrayList<>();
        for(int i = 0; i < d.numExamples(); i++)
            next.add(d.get(i).copy());
        Collections.shuffle(next);
        JavaRDD<DataSet> data = sc.parallelize(next);

        MultiLayerNetwork network = master.fitDataSet(data);

        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(),network.output(d.getFeatureMatrix()));
        System.out.println("Averaged once " + evaluation.stats());


        String json = conf.toJson();
        FileUtils.writeStringToFile(new File(app.outputPath + ".json"),json);
        FileUtils.writeStringToFile(new File(app.outputPath + ".params"), StringUtils.join(network.params().data().asDouble(),','));


    }

}
