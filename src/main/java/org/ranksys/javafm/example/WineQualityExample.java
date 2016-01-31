/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.example;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import static java.lang.Double.parseDouble;
import java.net.MalformedURLException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.IntToDoubleFunction;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.ArrayListFMData;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.data.MatrixFMData;
import org.ranksys.javafm.instance.FMInstance;
import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.learner.sgd.RMSEFMLearner;
import static java.lang.Math.max;
import static java.lang.Math.min;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import static java.util.stream.Collectors.groupingBy;

/**
 * Regression example with the Wine Quality dataset.<br>
 * 
 * https://archive.ics.uci.edu/ml/datasets/Wine+Quality
 *
 * @author Saúl Vargas (Saul@VargasSandoval.es)
 */
public class WineQualityExample {

    public static void main(String[] args) throws Exception {
        FMData<FMInstance> dataset = getWineQualityDataset();
        List<FMData<FMInstance>> partition = getRandomPartition(dataset, 0.6, new Random(1L));
        FMData<FMInstance> train = partition.get(0);
        FMData<FMInstance> test = partition.get(1);

        double alpha = 0.01;
        double sample = 1.0;
        double lambdaB = 0.1;
        IntToDoubleFunction lambdaW = i -> 10.0;
        IntToDoubleFunction lambdaM = i -> 0.1;
        int K = 10;

        FMLearner<FMInstance> learner = new RMSEFMLearner(alpha, sample, lambdaB, lambdaW, lambdaM);

        FM<FMInstance> fm = learner.learn(K, train, test);

        System.out.println(fm.getB());
        System.out.println(fm.getW());
        System.out.println(fm.getM());
    }

    private static FMData<FMInstance> getWineQualityDataset() throws MalformedURLException, IOException {
        int columns = 11;
        int rows = 4898;

        DenseDoubleMatrix1D targets = new DenseDoubleMatrix1D(rows);
        SparseDoubleMatrix2D features = new SparseDoubleMatrix2D(rows, columns);

        String filePath = "winequality-white.csv";
        if (!new File(filePath).exists()) {
            URL url = new URL("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv");
            ReadableByteChannel rbc = Channels.newChannel(url.openStream());
            FileOutputStream fos = new FileOutputStream(filePath);
            fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
        }

        InputStream is = new FileInputStream(filePath);

        try (BufferedReader in = new BufferedReader(new InputStreamReader(is))) {
            in.readLine();
            String instance;
            int row = 0;
            while ((instance = in.readLine()) != null) {
                String[] tokens = instance.split(";");
                for (int col = 0; col < columns; col++) {
                    features.setQuick(row, col, parseDouble(tokens[col]));
                }
                targets.setQuick(row, parseDouble(tokens[columns]));
                row++;
            }
        }

        for (int col = 0; col < columns; col++) {
            DoubleMatrix1D column = features.viewColumn(col);
            double max = column.aggregate((x, y) -> max(x, y), x -> x);
            double min = column.aggregate((x, y) -> min(x, y), x -> x);

            if (max == min) {
                column.assign(0.0);
            } else {
                column.assign(x -> (x - min) / (max - min));
            }
        }

        return new MatrixFMData(targets, features);
    }

    private static List<FMData<FMInstance>> getRandomPartition(FMData<FMInstance> dataset, double trainProp, Random rnd) {
        Map<Boolean, List<FMInstance>> partition = dataset.stream()
                .collect(groupingBy(instance -> rnd.nextDouble() < trainProp));
        FMData<FMInstance> train = new ArrayListFMData<>(dataset.numFeatures(), partition.get(true));
        FMData<FMInstance> test = new ArrayListFMData<>(dataset.numFeatures(), partition.get(false));

        return Arrays.asList(train, test);
    }

}