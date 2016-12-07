package org.ranksys.javafm;

import java.util.Random;

import org.apache.commons.math3.util.FastMath;

/**
 * A factorization that ends with a logistic link function, forcing the
 * generated predictions to be in the range 0..1
 */
public class LogisticFM extends FM {
    public LogisticFM(double b, double[] w, double[][] m) {
        super(b, w, m);
    }

    public LogisticFM(int numFeatures, int K, Random rnd, double sdev) {
        super(numFeatures, K, rnd, sdev);
    }

    @Override
    public double predict(FMInstance x) {
        double z = super.predict(x);
        return 1 / (1 + FastMath.exp(-z));
    }
}
