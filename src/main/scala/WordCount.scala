import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangfan on 17/4/23.
  */
object WordCount {
  def main(args: Array[String]): Unit ={
    val conf = new SparkConf().setAppName("WordCount")
    var sc = new SparkContext(conf)
    var rdd = sc.textFile("file:///Users/zhangfan/kaifa/spark/spark-2.1.0-bin-hadoop2.7/LICENSE")
    val wordcount = rdd.flatMap(_.split(" ")).map(x=>(x,1)).reduceByKey(_+_)
    val wordsort = wordcount.map(x=>(x._2,x._1)).sortByKey(false).map(x=>(x._2,x._1))
    wordsort.saveAsTextFile("file:///Users/zhangfan/kaifa/spark/wordsort")
  }
}
