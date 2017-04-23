package chapter01

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

/**
  * A simple Spark app in Scala
  */
object ScalaApp {

  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "First Spark App")

    // we take the raw data in CSV format and convert it into a set of records of the form (user, product, price)
    val data = sc.textFile("file:///Users/zhangfan/kaifa/IDEA/sparklearning/src/main/scala/chapter01/data/UserPurchaseHistory.csv")
      .map(line => line.split(","))
      .map(purchaseRecord => (purchaseRecord(0), purchaseRecord(1), purchaseRecord(2)))

    // 求购买数  let's count the number of purchases
    val numPurchases = data.count()

    // 求有多少个不同客户购买过商品  let's count how many unique users made purchases
    val uniqueUsers = data.map { case (user, product, price) => user }.distinct().count()

    // 求和得出总收入 let's sum up our total revenue
    val totalRevenue = data.map { case (user, product, price) => price.toDouble }.sum()

    // 求最畅销的产品是什么 let's find our most popular product
    val productsByPopularity = data
      .map { case (user, product, price) => (product, 1) }
      .reduceByKey(_ + _)
      .collect()
      .sortBy(-_._2)
    val mostPopular = productsByPopularity(0)

    // finally, print everything out
    println("Total purchases: " + numPurchases)
    println("Unique users: " + uniqueUsers)
    println("Total revenue: " + totalRevenue)
    println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))

    sc.stop()
  }

}

