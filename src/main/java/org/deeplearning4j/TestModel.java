package org.deeplearning4j;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

/**
 * Created by agibsonccc on 2/2/15.
 */
public class TestModel {

    public static void main(String[] args) throws Exception {
        DataSet d = new MnistDataSetIterator(60000,60000).next();
        MultiLayerNetwork averagedOnce = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json"))));

        INDArray params2 = Nd4j.readTxt("params.txt",",");
        averagedOnce.init();
        averagedOnce.setParameters(params2);
        //MultiLayerNetwork averagedPerIteration = SerializationUtils.readObject(new File("net.averagedperiteration.ser"));

        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(),averagedOnce.output(d.getFeatureMatrix()));
        System.out.println("Averaged once " + evaluation.stats());

/*
        Evaluation evaluation2 = new Evaluation();
        evaluation2.eval(d.getLabels(),averagedPerIteration.output(d.getFeatureMatrix()));
        System.out.println("Averaged once " + evaluation2.stats());*/
    }


}
