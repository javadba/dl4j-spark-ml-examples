package org.deeplearning4j.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2Vec;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecPerformer;
import org.deeplearning4j.util.SerializationUtils;
import org.springframework.core.io.ClassPathResource;

import java.io.File;

/**
 * Created by agibsonccc on 3/30/15.
 */
public class Word2VecExample {

    public static void main(String[] args) throws Exception {
        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[8]").set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, "false").set(Word2VecPerformer.NEGATIVE,String.valueOf("0"))
                .set("spark.akka.frameSize", "100").set(Word2VecPerformer.VECTOR_LENGTH,String.valueOf(300))
                .setAppName("mnist");

        System.out.println("Setting up Spark Context...");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> rdd = sc.textFile(new File(args[0]).toURI().toString());
        Word2Vec vec = new Word2Vec();
        Pair<VocabCache,WeightLookupTable> table = vec.train(rdd);
        SerializationUtils.saveObject(table,new File("table.ser"));
        System.out.println("Saved model");

    }

}
