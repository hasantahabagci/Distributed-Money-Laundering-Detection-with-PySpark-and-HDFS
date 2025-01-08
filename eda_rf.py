from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, hour, date_format, to_timestamp
import matplotlib.pyplot as plt
import seaborn as sns
import time

spark = SparkSession.builder \
    .master("yarn") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "2") \
    .appName("My Dataproc Notebook") \
    .getOrCreate()

# Veri yükleme
start = time.time()
df = spark.read.csv("gs://my-data-bucket-aydne-2024/HI-Medium_Trans.csv", header=True, inferSchema=True)
stop = time.time()
print('Df load time:', stop - start)

# Veri gösterme
start = time.time()
df.show(5)
stop = time.time()
print('Df show time:', stop - start)
#Df load time: 40.09629726409912

# VERİNİN 10 DA 1 İNİ KULLANCAKSAN, BİZ ÖYLE YAPTIK
#df = df.sample(fraction=0.1, seed=42)  # Use 10% of the data

df = df.dropDuplicates()

# df_model = df.drop('Timestamp', 'Account2', 'Account4', 'Date', 'Day', 'Time', 'Hour Block')
df_model = df.drop('Timestamp', 'Account2', 'Account4')
df_model
label_col = 'Is Laundering'
numerical_cols = []  # No numerical columns to use as-is after binning
categorical_cols = ['From Bank', 'To Bank', 'Receiving Currency', 'Payment Currency', 'Payment Format']
binning_cols = ['Amount Received', 'Amount Paid']

from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    OneHotEncoder,
    QuantileDiscretizer
)
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    RandomForestClassifier,
    MultilayerPerceptronClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SparkModelTrainer:
    def __init__(
        self,
        label_col,
        numerical_cols=None,
        categorical_cols=None,
        binning_cols=None,
        num_buckets=5,
        spark=None
    ):
        """
        Initializes the SparkModelTrainer with label, numerical, categorical, and binning columns.

        :param label_col: The name of the label column.
        :param numerical_cols: A list of numerical feature column names to be used as-is.
        :param categorical_cols: A list of categorical feature column names.
        :param binning_cols: A list of numerical feature column names to be discretized into bins.
        :param num_buckets: Number of buckets to create for each binning column.
        :param spark: Existing SparkSession. If None, a new one is created.
        """
        self.label_col = label_col
        self.numerical_cols = numerical_cols if numerical_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.binning_cols = binning_cols if binning_cols else []
        self.num_buckets = num_buckets
        self.spark = spark or SparkSession.builder \
            .appName("SparkModelTrainer") \
            .getOrCreate()
        
        # Initialize stages for the pipeline
        self.stages = []
        
        # Handle binning of numerical columns
        if self.binning_cols:
            self.discretizers = []
            for col in self.binning_cols:
                discretizer = QuantileDiscretizer(
                    inputCol=col,
                    outputCol=f"{col}_binned",
                    numBuckets=self.num_buckets,
                    relativeError=0.0
                )
                self.discretizers.append(discretizer)
                self.stages.append(discretizer)
            
            # After binning, treat the binned columns as categorical
            self.categorical_cols += [f"{col}_binned" for col in self.binning_cols]
        
        # Handle categorical columns: StringIndexer + OneHotEncoder
        if self.categorical_cols:
            self.indexers = [
                StringIndexer(
                    inputCol=col,
                    outputCol=f"{col}_indexed",
                    handleInvalid='keep'
                ) for col in self.categorical_cols
            ]
            self.stages += self.indexers
            
            self.encoders = [
                OneHotEncoder(
                    inputCol=f"{col}_indexed",
                    outputCol=f"{col}_encoded",
                    dropLast=True  # To avoid the dummy variable trap
                ) for col in self.categorical_cols
            ]
            self.stages += self.encoders
        
        # Assemble all features into a single vector
        assembler_input = []
        
        # Add numerical columns (if any)
        if self.numerical_cols:
            assembler_input += self.numerical_cols
        
        # Add encoded categorical columns
        if self.categorical_cols:
            assembler_input += [f"{col}_encoded" for col in self.categorical_cols]
        
        self.assembler = VectorAssembler(
            inputCols=assembler_input,
            outputCol="features"
        )
        self.stages.append(self.assembler)
        
        # Initialize Pipeline
        self.pipeline = Pipeline(stages=self.stages)
        
        # Placeholders for models
        self.rf_model = None
        self.ann_model = None
        
        # Dictionary to store evaluation metrics
        self.metrics = {}
        
    def prepare_data(self, df, test_ratio=0.2, seed=42):
        """
        Prepares and splits the data into training and testing sets, applying preprocessing.

        :param df: Input Spark DataFrame.
        :param test_ratio: Proportion of the dataset to include in the test split.
        :param seed: Random seed.
        :return: Tuple of (training DataFrame, testing DataFrame)
        """
        # **Verify Data Types Before Transformation**
        print("Schema before transformation:")
        df.printSchema()
        
        # Apply preprocessing pipeline
        preprocessed_model = self.pipeline.fit(df)
        preprocessed_df = preprocessed_model.transform(df)
        
        # **Verify Schema After Transformation**
        print("Schema after transformation:")
        preprocessed_df.printSchema()
        
        # Select the features and label
        assembled_df = preprocessed_df.select("features", self.label_col)
        
        # Split the data
        train_df, test_df = assembled_df.randomSplit([1 - test_ratio, test_ratio], seed=seed)
        print(f"Training Data Count: {train_df.count()}")
        print(f"Testing Data Count: {test_df.count()}")
        return train_df, test_df

    def train_random_forest(self, train_df, num_trees=100, max_depth=5):
        """
        Trains a Random Forest classifier.

        :param train_df: Training DataFrame.
        :param num_trees: Number of trees in the forest.
        :param max_depth: Maximum depth of the trees.
        """
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=self.label_col,
            numTrees=num_trees,
            maxDepth=max_depth,
            seed=42
        )
        self.rf_model = rf.fit(train_df)
        print("Random Forest model trained.")

    def train_ann(self, train_df, layers=None, max_iter=100):
        """
        Trains an Artificial Neural Network classifier.

        :param train_df: Training DataFrame.
        :param layers: List specifying the number of nodes in each layer.
                       If None, it will be dynamically determined.
        :param max_iter: Maximum number of iterations.
        """
        if layers is None:
            # Dynamically determine input layer size
            feature_size = len(train_df.first()["features"])
            layers = [feature_size, 20, 10, 2]  # Example: [input, hidden1, hidden2, output]
            print(f"Dynamic ANN layers set to: {layers}")
        ann = MultilayerPerceptronClassifier(
            featuresCol="features",
            labelCol=self.label_col,
            layers=layers,
            maxIter=max_iter,
            blockSize=128,
            seed=42
        )
        self.ann_model = ann.fit(train_df)
        print("ANN model trained.")

    def predict(self, model_name, test_df):
        """
        Makes predictions using the specified model.

        :param model_name: Name of the model ('random_forest', 'ann').
        :param test_df: Testing DataFrame.
        :return: DataFrame with predictions.
        """
        if model_name == 'random_forest':
            if not self.rf_model:
                raise ValueError("Random Forest model is not trained.")
            return self.rf_model.transform(test_df)
        elif model_name == 'ann':
            if not self.ann_model:
                raise ValueError("ANN model is not trained.")
            return self.ann_model.transform(test_df)
        else:
            raise ValueError("Unsupported model name. Choose from 'random_forest', 'ann'.")

    def evaluate(self, model_name, predictions, metric="accuracy"):
        """
        Evaluates the predictions using the specified metric.

        :param model_name: Name of the model ('random_forest', 'ann').
        :param predictions: DataFrame with predictions and label.
        :param metric: Evaluation metric ('accuracy', 'f1', 'auc').
        :return: Evaluation score.
        """
        if metric in ["accuracy", "f1"]:
            evaluator = MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol="prediction",
                metricName=metric
            )
        elif metric == "auc":
            evaluator = BinaryClassificationEvaluator(
                labelCol=self.label_col,
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
        else:
            raise ValueError("Unsupported metric. Choose from 'accuracy', 'f1', 'auc'.")
        
        score = evaluator.evaluate(predictions)
        # Store the metric
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
        self.metrics[model_name][metric] = score
        return score

    def plot_metrics(self, metric="accuracy"):
        """
        Plots the specified evaluation metric for all trained models.

        :param metric: The metric to plot ('accuracy', 'f1', 'auc').
        """
        # Prepare data for plotting
        models = []
        scores = []
        for model_name, metrics in self.metrics.items():
            if metric in metrics:
                models.append(model_name)
                scores.append(metrics[metric])
        
        if not models:
            print(f"No evaluations found for metric: {metric}")
            return
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Model': models,
            metric.capitalize(): scores
        })
        
        # Set plot style
        sns.set(style="whitegrid")
        
        # Initialize the matplotlib figure
        plt.figure(figsize=(10, 6))
        
        # Create a barplot
        sns.barplot(x='Model', y=metric.capitalize(), data=plot_df, palette="viridis")
        
        # Add titles and labels
        plt.title(f'Comparison of Models based on {metric.capitalize()}', fontsize=16)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.xlabel('Model', fontsize=14)
        
        # Annotate bars with scores
        for index, row in plot_df.iterrows():
            plt.text(index, row[metric.capitalize()] + 0.01, f"{row[metric.capitalize()]:.4f}", 
                     color='black', ha="center", fontsize=12)
        
        plt.ylim(0, 1)  # Assuming metrics are between 0 and 1
        plt.show()

    def plot_all_metrics(self):
        """
        Plots all available evaluation metrics for all trained models.
        """
        available_metrics = set()
        for metrics in self.metrics.values():
            available_metrics.update(metrics.keys())
        
        for metric in available_metrics:
            self.plot_metrics(metric=metric)

    def stop_spark(self):
        """Stops the Spark session."""
        self.spark.stop()
        print("Spark session stopped.")

trainer = SparkModelTrainer(
    label_col=label_col,
    numerical_cols=numerical_cols,  # Empty list since all numerical columns are binned
    categorical_cols=categorical_cols,  # Initial categorical columns
    binning_cols=binning_cols,  # Columns to discretize
    num_buckets=5,  # Number of bins for discretization
    spark=spark
)

train_df, test_df = trainer.prepare_data(df_model, test_ratio=0.2, seed=42)

# **Train Random Forest:**
trainer.train_random_forest(train_df, num_trees=3, max_depth=5)

train_df = train_df.repartition(32)  # Partition sayısını artırın

rf_predictions = trainer.predict('random_forest', test_df)
rf_accuracy = trainer.evaluate('random_forest', rf_predictions, metric="accuracy")
rf_f1 = trainer.evaluate('random_forest', rf_predictions, metric="f1")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Random Forest F1 Score: {rf_f1:.4f}")
