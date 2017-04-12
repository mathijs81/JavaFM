/*
 * Copyright (C) 2016 RankSys http://ranksys.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.ranksys.javafm.learner.gd;

import java.util.logging.Logger;

import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.learner.FMLearner;

/**
 * Stochastic gradient descent learner with momentum, derived from
 * PointWiseGradientDescent
 *
 * @author Mathijs Vogelzang
 */
public class MomentumPointWiseGradientDescent implements FMLearner<FMData> {

    private static final Logger LOG = Logger.getLogger(PointWiseGradientDescent.class.getName());

    private final double learnRate;
    private final int numIter;
    private final PointWiseError error;
    private final double regB;
    private final double[] regW;
    private final double[] regM;
    private final double momentum;

    public MomentumPointWiseGradientDescent(double learnRate, double momentum, int numIter, PointWiseError error, double regB, double[] regW, double[] regM) {
        this.learnRate = learnRate;
        this.momentum = momentum;
        this.numIter = numIter;
        this.error = error;
        this.regB = regB;
        this.regW = regW;
        this.regM = regM;
    }

    @Override
    public double error(FM fm, FMData test) {
        return test.stream()
                .mapToDouble(x -> error.error(fm, x))
                .average().getAsDouble();
    }

    private double velocityB = 0;
    private double[] velocityW;
    private double[][] velocityM;

    @Override
    public void learn(FM fm, FMData train, FMData test) {
        LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));

        velocityW = new double[fm.getW().length];
        int mDimension1 = fm.getM().length;
        int mDimension2 = fm.getM()[0].length;
        velocityM = new double[mDimension1][mDimension2];

        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();

            train.shuffle();

            train.stream().forEach(x -> {
                double b = fm.getB();
                double[] w = fm.getW();
                double[][] m = fm.getM();

                double lambda = error.dError(fm, x);

                velocityB = velocityB * momentum + (lambda + regB * b);
                fm.setB(b - learnRate * velocityB);

                double[] xm = new double[m[0].length];
                x.consume((i, xi) -> {
                    for (int j = 0; j < xm.length; j++) {
                        xm[j] += xi * m[i][j];
                    }
                    velocityW[i] = velocityW[i] * momentum + lambda * xi + regW[i] * w[i];
                    w[i] -= learnRate * velocityW[i];
                });

                x.consume((i, xi) -> {
                    for (int j = 0; j < m[i].length; j++) {
                        velocityM[i][j] = velocityM[i][j] * momentum + lambda * xi * xm[j]
                            - lambda * xi * xi * m[i][j] + regM[i] * m[i][j];
                        m[i][j] -= learnRate * velocityM[i][j];
                    }
                });
            });

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.info(String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }

        velocityW = null;
        velocityM = null;
    }
}