import java.io.File;

import org.jfree.chart.labels.StandardCategoryItemLabelGenerator;
import org.jfree.chart.renderer.category.StandardBarPainter;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.converters.CSVLoader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.meta.Stacking;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.REPTree;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.LMT;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GradientPaint;
import java.awt.GridLayout;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.List;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.CategoryItemRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

public class WekaIrisMultiple {
    public static void main(String[] args) throws Exception {
        // Cargar conjunto de datos Iris en formato CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("iris.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Dividir datos en conjunto de entrenamiento y prueba (80/20)
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        // Cargar todos los clasificadores disponibles
        WekaPackageManager.loadPackages(false, true, false);
        Classifier[] classifiers = {
                new J48(), new RandomCommittee(), new RandomForest(),
                new REPTree(), new Bagging(),
                new LogitBoost(),
                new RandomSubSpace(), new AdaBoostM1(),
                new DecisionTable(), new JRip(), new OneR(),
                new PART(), new Logistic(), new MultilayerPerceptron(),
                new IBk(), new LMT(), new Stacking(), new NaiveBayes(),
                new SMO()
        };

        // Realizar entrenamiento y evaluación para cada clasificador
        List<String> classifierNames = new ArrayList<>();
        List<Double> accuracies = new ArrayList<>();
        for (Classifier classifier : classifiers) {
            classifier.buildClassifier(train);
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(classifier, test);
            System.out.println("Clasificador: " + classifier.getClass().getSimpleName());
            System.out.println(eval.toSummaryString("\nResultados de la evaluación:\n", false));
            classifierNames.add(classifier.getClass().getSimpleName());
            accuracies.add(eval.pctCorrect());
        }

        // Crear dataset con los resultados de evaluación
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < classifierNames.size(); i++) {
            dataset.addValue(accuracies.get(i), classifierNames.get(i), "Accuracy");
        }

        // Crear gráfico de barras con los resultados de evaluación
        JFreeChart chart = ChartFactory.createBarChart(
                "Resultados de evaluación de clasificadores", // Título del gráfico
                "Clasificador", // Etiqueta del eje X
                "Accuracy", // Etiqueta del eje Y
                dataset, // Dataset con los resultados de evaluación
                PlotOrientation.VERTICAL, // Orientación del gráfico
                true, // Mostrar leyenda
                true, // Mostrar tooltips
                false // No mostrar urls
        );

        // Personalizar gráfico de barras
        chart.setBackgroundPaint(Color.WHITE);
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.BLACK);
        CategoryAxis xAxis = plot.getDomainAxis();
        xAxis.setCategoryLabelPositions(CategoryLabelPositions.createUpRotationLabelPositions(Math.PI / 6.0));
        CategoryItemRenderer renderer = plot.getRenderer();
        renderer.setDefaultItemLabelGenerator(new StandardCategoryItemLabelGenerator());
        BarRenderer barRenderer = (BarRenderer) renderer;
        barRenderer.setBarPainter(new StandardBarPainter());
        Paint[] colors = new Paint[classifierNames.size()];
        for (int i = 0; i < colors.length; i++) {
            colors[i] = new GradientPaint(0.0f, 0.0f, Color.getHSBColor((float) i / (float) colors.length, 1.0f, 1.0f),
                    0.0f, 0.0f, new Color(240, 240, 240));
        }
        barRenderer.setSeriesPaint(0, colors[0]);
        barRenderer.setSeriesPaint(1, colors[1]);
        barRenderer.setSeriesPaint(2, colors[2]);
        barRenderer.setSeriesPaint(3, colors[3]);
        barRenderer.setSeriesPaint(4, colors[4]);
        barRenderer.setSeriesPaint(5, colors[5]);
        Shape shape = new Ellipse2D.Double(-3.0, -3.0, 6.0, 6.0);
        for (int i = 0; i < colors.length; i++) {
            barRenderer.setSeriesShape(i, shape);
            barRenderer.setSeriesPaint(i, colors[i]);
        }

        // Mostrar gráfico de barras en una ventana
        ApplicationFrame frame = new ApplicationFrame("Resultados de evaluación de clasificadores");
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(800, 600));
        frame.setContentPane(chartPanel);
        frame.pack();
        RefineryUtilities.centerFrameOnScreen(frame);
        frame.setVisible(true);
    }
}

