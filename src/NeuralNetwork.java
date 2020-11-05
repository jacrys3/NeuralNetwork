import java.util.HashMap;
import java.util.List;

public class NeuralNetwork {
    Matrix weightsIn, weightsOut, biasHid, biasOut;
    double lRate = 0.01;

    public NeuralNetwork(int i, int h, int o){
        weightsIn = new Matrix(h, i);
        weightsOut = new Matrix(o, h);

        biasHid = new Matrix(h, 1);
        biasOut = new Matrix(o, 1);
    }

    public List<Double> predict(double[] x){
        Matrix input = Matrix.fromArray(x);
        Matrix hidden = Matrix.multiply(weightsIn, input);
        hidden.add(biasHid);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weightsOut, hidden);
        output.add(biasOut);
        output.sigmoid();

        return output.toArray();
    }

    public void train(double[] x, double[] y){
        Matrix input = Matrix.fromArray(x);
        Matrix hidden = Matrix.multiply(weightsIn, input);
        hidden.add(biasHid);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weightsOut, hidden);
        output.add(biasOut);
        output.sigmoid();

        Matrix target = Matrix.fromArray(y);

        Matrix error = Matrix.subtract(target, output);
        Matrix gradient = output.dsigmoid();
        gradient.multiply(error);
        gradient.multiply(lRate);

        Matrix hiddenT = Matrix.transpose(hidden);
        Matrix whoDelta = Matrix.multiply(gradient, hiddenT);

        weightsOut.add(whoDelta);
        biasOut.add(gradient);

        Matrix whoT = Matrix.transpose(weightsOut);
        Matrix hiddenErrors = Matrix.multiply(whoT, error);

        Matrix hGradient = hidden.dsigmoid();
        hGradient.multiply(hiddenErrors);
        hGradient.multiply(lRate);

        Matrix inT = Matrix.transpose(input);
        Matrix winDelta = Matrix.multiply(hGradient, inT);

        weightsIn.add(winDelta);
        biasHid.add(hGradient);
    }

    public void fit(double[][] x, double[][] y, int runs){
        for (int i = 0; i < runs; i++) {
            int sampleN = (int)(Math.random() * x.length);
            this.train(x[sampleN], y[sampleN]);
        }
    }
}
