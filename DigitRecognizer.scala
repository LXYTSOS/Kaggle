package com.lxy.kaggle.digitrecognizer

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.DecisionTree

/**
 * @author sl169
 */
object DigitRecognizer {
  def main(args: Array[String]){
    val conf = new SparkConf()
    .setAppName("DigitRecgonizer")
    .setMaster("local[*]")
    .set("spark.driver.memory", "10G")
    val sc = new SparkContext(conf)
    
    val rawData = sc.textFile("file:///home/shdxspark/suisui/train-noheader.csv")
    val records = rawData.map(line => line.split(","))
    val data = records.map{ r =>
      val label = r(0).toInt
      val features = r.slice(1, r.size).map(p => p.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    
//    val nbModel = NaiveBayes.train(data)
    
//    val nbTotalCorrect = data.map { point =>
//      if (nbModel.predict(point.features) == point.label) 1 else 0
//    }.sum
//    val numData = data.count()
//    println(numData)
    //42000
//    val nbAccuracy = nbTotalCorrect / numData
//    println("准确率："+nbAccuracy)
    //准确率：0.8261190476190476 
    
    val unlabeledData = sc.textFile("file:///home/shdxspark/suisui/test-noheader.csv")
    val unlabeledRecords = unlabeledData.map(line => line.split(","))
    val features = unlabeledRecords.map{ r =>
      val f = r.map(p => p.toDouble)
      Vectors.dense(f)
    }
//    val num = unlabeledData.count()
    
//    val predictions = nbModel.predict(features).map { p => p.toInt }
//    val raw = unlabeledData.collect()
//    val pre = predictions.collect()
//    val p = (0 to num.toInt - 1).map{ i =>
//      val label = pre(i).toString()
//      val feature = raw(i)
//      label+","+feature
//    }
//    val result = sc.parallelize(p)
//    result.repartition(1).saveAsTextFile("file:///home/shdxspark/suisui/digitRec.txt")
    
    //随机森林模型
//    val numClasses = 10
//    val categoricalFeaturesInfo = Map[Int, Int]()
//    val numTrees = 70 
//    val featureSubsetStrategy = "auto" 
//    val impurity = "gini"
//    val maxDepth = 30
//    val maxBins = 32
//    val randomForestModel = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    
//    val nbTotalCorrect = data.map { point =>
//      if (randomForestModel.predict(point.features) == point.label) 1 else 0
//    }.sum
//    val numData = data.count()
//    println(numData)
//    //42000
//    val nbAccuracy = nbTotalCorrect / numData
//    println("准确率："+nbAccuracy)
    //numTree=3,maxDepth=4,准确率：0.5507619047619048
    //numTree=4,maxDepth=5,准确率：0.7023095238095238
    //numTree=5,maxDepth=6,准确率：0.693595238095238
    //numTree=6,maxDepth=7,准确率：0.8426428571428571
    //numTree=7,maxDepth=8,准确率：0.879452380952381
    //numTree=8,maxDepth=9,准确率：0.9105714285714286
    //numTree=9,maxDepth=10,准确率：0.9446428571428571
    //numTree=10,maxDepth=11,准确率：0.9611428571428572
    //numTree=11,maxDepth=12,准确率：0.9765952380952381
    //numTree=12,maxDepth=13,准确率：0.9859523809523809
    //numTree=13,maxDepth=14,准确率：0.9928333333333333
    //numTree=14,maxDepth=15,准确率：0.9955
    //numTree=15,maxDepth=16,准确率：0.9972857142857143
    //numTree=16,maxDepth=17,准确率：0.9979285714285714
    //numTree=17,maxDepth=18,准确率：0.9983809523809524
    //numTree=18,maxDepth=19,准确率：0.9989285714285714
    //numTree=19,maxDepth=20,准确率：0.9989523809523809
    //numTree=20,maxDepth=21,准确率：0.999
    //numTree=21,maxDepth=22,准确率：0.9994761904761905
    //numTree=22,maxDepth=23,准确率：0.9994761904761905
    //numTree=23,maxDepth=24,准确率：0.9997619047619047
    //numTree=24,maxDepth=25,准确率：0.9997857142857143
    //numTree=25,maxDepth=26,准确率：0.9998333333333334
    //numTree=29,maxDepth=30,准确率：0.9999523809523809
    
//    val predictions = randomForestModel.predict(features).map { p => p.toInt }
//    predictions.repartition(1).saveAsTextFile("file:///home/shdxspark/suisui/digitRec.txt")
    
    //决策树
    val numClasses = 10
    val categoricalFeaturesInfo = Map[Int, Int](4 -> 10)
    val impurity = "gini"
    val maxDepth = 30
    val maxBins = 128
    
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    
    val decisionTreeModel = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,impurity, maxDepth, maxBins)
    val nbTotalCorrect = testData.map { point =>
      if (decisionTreeModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val numData = testData.count()
    println(numData)
    val nbAccuracy = nbTotalCorrect / numData
    println("准确率："+nbAccuracy)
    //确率：0.9986428571428572,在训练数据上计算的，没使用交叉验证
    //准确率：0.845889590157386,交叉验证0.7,0.3，maxDepth = 30,maxBins = 32
    //准确率：0.8478972149070305,交叉验证0.7,0.3，maxDepth = 30,maxBins = 64
    //准确率：0.8492602958816473,交叉验证0.7,0.3，maxDepth = 30,maxBins = 128
    //准确率：0.8513740886146943
    

  }
}