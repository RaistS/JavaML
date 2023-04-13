import java.io.File;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class WekaIris {
    public static void main(String[] args) throws Exception {
        // Cargar conjunto de datos Iris en formato CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("iris.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Aleatorizar datos
        Randomize randFilter = new Randomize();
        randFilter.setInputFormat(data);
        data = Filter.useFilter(data, randFilter);

        // Dividir datos en conjunto de entrenamiento y prueba
        RemovePercentage splitFilter = new RemovePercentage();
        splitFilter.setInputFormat(data);
        splitFilter.setPercentage(20);
        Instances test = Filter.useFilter(data, splitFilter);
        splitFilter.setInvertSelection(true);
        Instances train = Filter.useFilter(data, splitFilter);

        // Entrenar modelo utilizando el algoritmo Naive Bayes
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);

        // Evaluar modelo utilizando conjunto de prueba
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(nb, test);

        // Mostrar resultados de la evaluación
        System.out.println(eval.toSummaryString("\nResultados de la evaluación:\n", false));
    }
}
