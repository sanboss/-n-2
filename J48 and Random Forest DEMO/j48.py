import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.core.converters import Loader
import weka.plot.graph as graph
import weka.plot.classifiers as plotcls
import weka.core.types as types

jvm.start()
data_location = "C:/Users/Huyvo/Downloads/"
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(data_location + "iris.arff")
data.class_is_last()
# classifier = Classifier(classname="weka.classifiers.trees.J48")
# classifier.set_property("confidenceFactor",types.double_to_float(0.3))
classifier = Classifier(classname="weka.classifiers.trees.J48",
                        options=["-C", "0.3"])
classifier.build_classifier(data)
evaluation = Evaluation(data)
#evaluation.crossvalidate_model(classifier,data,10,Random(42))
evaluation.evaluate_train_test_split(classifier, data,30, Random(1))
print(classifier)
print(classifier.graph)
print(evaluation.summary())
print("pctCorrect: " + str(evaluation.percent_correct))
print("incorrect: " + str(evaluation.incorrect))
print(evaluation.class_details())
print (evaluation.matrix())
plotcls.plot_classifier_errors(evaluation.predictions,absolute = False, wait =True)
graph.plot_dot_graph(classifier.graph)