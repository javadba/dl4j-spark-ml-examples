package org.deeplearning4j.word2vec;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.util.SerializationUtils;

import java.io.File;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class CustomWord2VecExample {


    public static void main(String[] args) throws Exception {
       Pair<VocabCache,WeightLookupTable> pair = SerializationUtils.readObject(new File(args[0]));



       Tsne tsne = new Tsne.Builder().setMaxIter(1000).stopLyingIteration(250)
                .learningRate(500).useAdaGrad(false).setMomentum(0.5)
                .normalize(true).usePca(false).build();

        pair.getSecond().plotVocab(tsne);




    }

}
