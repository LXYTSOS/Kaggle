package com.lxy.kaggle.digitrecognizer

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.NaiveBayes

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
    
    val nbModel = NaiveBayes.train(data)
    
//     val nbTotalCorrect = data.map { point =>
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
    val featrues = unlabeledRecords.map{ r =>
      val f = r.map(p => p.toDouble)
      Vectors.dense(f)
    }
    val num = unlabeledData.count()
    
    val predictions = nbModel.predict(featrues).map { p => p.toInt }
    val raw = unlabeledData.collect()
    val pre = predictions.collect()
    val p = (0 to num.toInt - 1).map{ i =>
      val label = pre(i).toString()
      val feature = raw(i)
      label+","+feature
    }
    val result = sc.parallelize(p)
    result.repartition(1).saveAsTextFile("file:///home/shdxspark/suisui/digitRec.txt")
  }
}