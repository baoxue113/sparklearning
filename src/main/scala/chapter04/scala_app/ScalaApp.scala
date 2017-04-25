package chapter04.scala_app

import org.apache.spark.SparkContext

/**
  * A simple Spark app in Scala
  */
object ScalaApp {

  def main(args: Array[String]) {
    /*
	This code is intended to be run in the Scala shell.
	Launch the Scala Spark shell by running ./bin/spark-shell from the Spark directory.
	You can enter each line in the shell and see the result immediately.
	The expected output in the Spark console is presented as commented lines following the
	relevant code

	The Scala shell creates a SparkContex variable available to us as 'sc'

  Ensure you start the shell with sufficient memory: ./bin/spark-shell --driver-memory 4g
*/
    // 加载SparcContext,每个虚拟机只能加载一个
    val sc = new SparkContext("local[2]", "First Spark App")
    /* Load the raw ratings data from a file. Replace 'PATH' with the path to the MovieLens data */
    val path = "file:///Users/zhangfan/kaifa/IDEA/sparklearning/src/main/scala/chapter04/data/ml-100k"
    val rawData = sc.textFile(path + "/u.data")
    val temp1 = rawData.first() //对数据解释为：用户id,电影id,评分,时间戳
    // 14/03/30 13:21:25 INFO SparkContext: Job finished: first at <console>:17, took 0.002843 s
    // res24: String = 196	242	3	881250949

    /* Extract the user id, movie id and rating only from the dataset */
    val rawRatings = rawData.map(_.split("\t").take(3)) //取每行的前3个数据
    val temp2 = rawRatings.first()
    // 14/03/30 13:22:44 INFO SparkContext: Job finished: first at <console>:21, took 0.003703 s
    // res25: Array[String] = Array(196, 242, 3)

    /* Import Spark's ALS recommendation model and inspect the train method */
    import org.apache.spark.mllib.recommendation.ALS // 导入ALS算法，最小二阶乘
//    ALS.train
    /*
      <console>:13: error: ambiguous reference to overloaded definition,
      both method train in object ALS of type (ratings: org.apache.spark.rdd.RDD[org.apache.spark.mllib.recommendation.Rating], rank: Int, iterations: Int)org.apache.spark.mllib.recommendation.MatrixFactorizationModel
      and  method train in object ALS of type (ratings: org.apache.spark.rdd.RDD[org.apache.spark.mllib.recommendation.Rating], rank: Int, iterations: Int, lambda: Double)org.apache.spark.mllib.recommendation.MatrixFactorizationModel
      match expected type ?
                    ALS.train
                        ^
    */

    /* Import the Rating class and inspect it */
    import org.apache.spark.mllib.recommendation.Rating
//    Rating()
    /*
      <console>:13: error: not enough arguments for method apply: (user: Int, product: Int, rating: Double)org.apache.spark.mllib.recommendation.Rating in object Rating.
      Unspecified value parameters user, product, rating.
                    Rating()
                          ^
    */


    /* Construct the RDD of Rating objects */
    // 将数据转成Rating对象集合
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    val temp3 = ratings.first()
    // 14/03/30 13:26:43 INFO SparkContext: Job finished: first at <console>:24, took 0.002808 s
    // res28: org.apache.spark.mllib.recommendation.Rating = Rating(196,242,3.0)

    /* Train the ALS model with rank=50, iterations=10, lambda=0.01 */
//     rank:对应ALS模型中的因子个数,也就是在低阶近似矩阵中的隐含特征个数。因子个 数一般越多越好。
//            但它也会直接影响模型训练和保存时所需的内存开销,尤其是在用户 和物品很多的时候。
//            因此实践中该参数常作为训练效果与系统开销之间的调节参数。通 常,其合理取值为10到200。
//     iterations:对应运行时的迭代次数。ALS能确保每次迭代都能降低评级矩阵的重建误 差,但一般经少数次迭代后
//                  ALS模型便已能收敛为一个比较合理的好模型。这样,大部分 情况下都没必要迭代太多次(10次左右
//                  一般就挺好)。
//     lambda:该参数控制模型的正则化过程,从而控制模型的过拟合情况。其值越高,正则 化越严厉。该参数的赋值与实际
//              数据的大小、特征和稀疏程度有关。和其他的机器学习 模型一样,正则参数应该通过用非样本的测试数据进
//              行交叉验证来调整。
//    作为示例,这里将使用的rank、iterations和lambda参数的值分别为50、10和0.01: val model = ALS.train(ratings, 50, 10, 0.01)
    val model = ALS.train(ratings, 50, 10, 0.01) //第一个参数 rank,第二个参数 iterations,第三个参数lambda
    // ...
    // 14/03/30 13:28:44 INFO MemoryStore: ensureFreeSpace(128) called with curMem=7544924, maxMem=311387750
    // 14/03/30 13:28:44 INFO MemoryStore: Block broadcast_120 stored as values to memory (estimated size 128.0 B, free 289.8 MB)
    // model: org.apache.spark.mllib.recommendation.MatrixFactorizationModel = org.apache.spark.mllib.recommendation.MatrixFactorizationModel@7c7fbd3b

    /* Inspect the user factors */
    val temp4 = model.userFeatures
    // res29: org.apache.spark.rdd.RDD[(Int, Array[Double])] = FlatMappedRDD[1099] at flatMap at ALS.scala:231

    /* Count user factors and force computation */
    val temp5 = model.userFeatures.count
    // ...
    // 14/03/30 13:30:08 INFO SparkContext: Job finished: count at <console>:26, took 5.009689 s
    // res30: Long = 943

    val temp6 = model.productFeatures.count
    // ...
    // 14/03/30 13:30:59 INFO SparkContext: Job finished: count at <console>:26, took 0.247783 s
    // res31: Long = 1682

    /* Make a prediction for a single user and movie pair */
    val predictedRating = model.predict(789, 123) //789用户，对电影123的预测评分。

    /* Make predictions for a single user across all movies */
    val userId = 789
    val k = 10
    val topKRecs = model.recommendProducts(userId, k) //为789用户推荐10个物品
    println(topKRecs.mkString("\n"))
    /*
    Rating(789,715,5.931851273771102)
    Rating(789,12,5.582301095666215)
    Rating(789,959,5.516272981542168)
    Rating(789,42,5.458065302395629)
    Rating(789,584,5.449949837103569)
    Rating(789,750,5.348768847643657)
    Rating(789,663,5.30832117499004)
    Rating(789,134,5.278933936827717)
    Rating(789,156,5.250959077906759)
    Rating(789,432,5.169863417126231)
    */

    /* Load movie titles to inspect the recommendations */
    val movies = sc.textFile(path + "/u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1))).collectAsMap()
    val temp7 = titles(123)
    // res68: String = Frighteners, The (1996)
    val temp8 = ratings.take(1)
    val moviesForUser = ratings.keyBy(_.user).lookup(789) //返回key为789用户的所有值 //lookup:http://blog.csdn.net/u012102306/article/details/51993548
    // moviesForUser: Seq[org.apache.spark.mllib.recommendation.Rating] = WrappedArray(Rating(789,1012,4.0), Rating(789,127,5.0), Rating(789,475,5.0), Rating(789,93,4.0), ...
    // ...
    println(moviesForUser.size)
    // 33 //将用户所看过的电影按评分排序，然后取前十个，并将电影id对应成名字
    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating)).foreach(println)
    val temp11 = moviesForUser.take(1) //关键在titles
    /*
    (Godfather, The (1972),5.0)
    (Trainspotting (1996),5.0)
    (Dead Man Walking (1995),5.0)
    (Star Wars (1977),5.0)
    (Swingers (1996),5.0)
    (Leaving Las Vegas (1995),5.0)
    (Bound (1996),5.0)
    (Fargo (1996),5.0)
    (Last Supper, The (1995),5.0)
    (Private Parts (1997),4.0)
    */
    val temp10 = topKRecs.take(1)
//    val temp9 = topKRecs.take(1).rating
    topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)
    /*
    (To Die For (1995),5.931851273771102)
    (Usual Suspects, The (1995),5.582301095666215)
    (Dazed and Confused (1993),5.516272981542168)
    (Clerks (1994),5.458065302395629)
    (Secret Garden, The (1993),5.449949837103569)
    (Amistad (1997),5.348768847643657)
    (Being There (1979),5.30832117499004)
    (Citizen Kane (1941),5.278933936827717)
    (Reservoir Dogs (1992),5.250959077906759)
    (Fantasia (1940),5.169863417126231)
    */

    /* Compute item-to-item similarities between an item and the other items */
    import org.jblas.DoubleMatrix
    val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
    // aMatrix: org.jblas.DoubleMatrix = [1.000000; 2.000000; 3.000000]

    /* Compute the cosine similarity between two vectors */
    def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
      //P87页
      //向量1与向量2的点积 / 各向量范数(或长度)的乘积
//      该相似度的取值在1到1之间。1表示完全相似,0表示两者互不相关(即无相似性)。这种衡 量方法很有帮助,
      // 因为它还能捕捉负相关性。也就是说,当为1时则不仅表示两者不相关,还表 示它们完全不同。
      vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
    }
    // cosineSimilarity: (vec1: org.jblas.DoubleMatrix, vec2: org.jblas.DoubleMatrix)Double
    val itemId = 567
    val itemFactor = model.productFeatures.lookup(itemId).head //以物品567为例从模型中取回其对应的因子
    // itemFactor: Array[Double] = Array(0.15179359424040248, -0.2775955241896113, 0.9886005994661484, ...
    val itemVector = new DoubleMatrix(itemFactor) //将因子做成向量
    // itemVector: org.jblas.DoubleMatrix = [0.151794; -0.277596; 0.988601; -0.464013; 0.188061; 0.090506; ...
    val temp12 = cosineSimilarity(itemVector, itemVector) //计算余弦相似度
    // res113: Double = 1.0000000000000002
    val sims = model.productFeatures.map{ case (id, factor) =>
      val factorVector = new DoubleMatrix(factor) //将每个特征点转成向量
      val sim = cosineSimilarity(factorVector, itemVector) //计算每个特征与567数据点的余弦相似度
      (id, sim)
    } // 对物品按照相似度排序,然后取出与物品567最相似的前10个物品
    val sortedSims = sims.top(k)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    // sortedSims: Array[(Int, Double)] = Array((567,1.0), (672,0.483244928887981), (1065,0.43267674923450905), ...
    println(sortedSims.mkString("\n"))
    /*
    (567,1.0000000000000002)
    (1471,0.6932331537649621)
    (670,0.6898690594544726)
    (201,0.6897964975027041)
    (343,0.6891221044611473)
    (563,0.6864214133620066)
    (294,0.6812075443259535)
    (413,0.6754663844488256)
    (184,0.6702643811753909)
    (109,0.6594872765176396)
    */

    /* We can check the movie title of our chosen movie and the most similar movies to it */
    println(titles(itemId))
    // Wes Craven's New Nightmare (1994)
    val sortedSims2 = sims.top(k + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    // slice:获取第1到第11个元素
    sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim) }.mkString("\n")
    /*
    (Hideaway (1995),0.6932331537649621)
    (Body Snatchers (1993),0.6898690594544726)
    (Evil Dead II (1987),0.6897964975027041)
    (Alien: Resurrection (1997),0.6891221044611473)
    (Stephen King's The Langoliers (1995),0.6864214133620066)
    (Liar Liar (1997),0.6812075443259535)
    (Tales from the Crypt Presents: Bordello of Blood (1996),0.6754663844488256)
    (Army of Darkness (1993),0.6702643811753909)
    (Mystery Science Theater 3000: The Movie (1996),0.6594872765176396)
    (Scream (1996),0.6538249646863378)
    */

    /* Compute squared error between a predicted and actual rating */
    // We'll take the first rating for our example user 789
    // 下面是验证模型的好坏
    val actualRating = moviesForUser.take(1)(0)
    // actualRating: Seq[org.apache.spark.mllib.recommendation.Rating] = WrappedArray(Rating(789,1012,4.0))
    //val predictedRating = model.predict(789, actualRating.product)
    // ...
    // 14/04/13 13:01:15 INFO SparkContext: Job finished: lookup at MatrixFactorizationModel.scala:46, took 0.025404 s
    // predictedRating: Double = 4.001005374200248

    //val squaredError = math.pow(predictedRating - actualRating.rating, 2.0)
    // squaredError: Double = 1.010777282523947E-6

    /* Compute Mean Squared Error across the dataset */
    // Below code is taken from the Apache Spark MLlib guide at: http://spark.apache.org/docs/latest/mllib-guide.html#collaborative-filtering-1
    val usersProducts = ratings.map{ case Rating(user, product, rating)  => (user, product)}
    // 预测的特征与预测值
    val predictions = model.predict(usersProducts).map{
      case Rating(user, product, rating) => ((user, product), rating)
    }
    // 真实的特征与预测值 将相同的(user, product)的rating连接起来
    val ratingsAndPredictions = ratings.map{
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions) // join -> http://blog.csdn.net/pmp4561705/article/details/53212196
    val MSE = ratingsAndPredictions.map{
      // (user, product):特征点，actual:预测的结果，predicted：实际的结果
      case ((user, product), (actual, predicted)) =>  math.pow((actual - predicted), 2)
    }.reduce(_ + _) / ratingsAndPredictions.count
    println("Mean Squared Error = " + MSE)
    // ...
    // 14/04/13 15:29:21 INFO SparkContext: Job finished: count at <console>:31, took 0.538683 s
    // Mean Squared Error = 0.08231947642632856
    val RMSE = math.sqrt(MSE) //开根号
    println("Root Mean Squared Error = " + RMSE)
    // Root Mean Squared Error = 0.28691370902473196

    /* Compute Mean Average Precision at K */

    /* Function to compute average precision given a set of actual and predicted ratings */
    // Code for this function is based on: https://github.com/benhamner/Metrics
    def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
      val predK = predicted.take(k) //取原始数据的10个数据
      var score = 0.0
      var numHits = 0.0
      for ((p, i) <- predK.zipWithIndex) {
        if (actual.contains(p)) {
          numHits += 1.0
          score += numHits / (i.toDouble + 1.0)
        }
      }
      if (actual.isEmpty) {
        1.0
      } else {
        score / scala.math.min(actual.size, k).toDouble
      }
    }
    val actualMovies = moviesForUser.map(_.product)
    // actualMovies: Seq[Int] = ArrayBuffer(1012, 127, 475, 93, 1161, 286, 293, 9, 50, 294, 181, 1, 1008, 508, 284, 1017, 137, 111, 742, 248, 249, 1007, 591, 150, 276, 151, 129, 100, 741, 288, 762, 628, 124)
    val predictedMovies = topKRecs.map(_.product)
    // predictedMovies: Array[Int] = Array(27, 497, 633, 827, 602, 849, 401, 584, 1035, 1014)
    val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
    // apk10: Double = 0.0

    /* Compute recommendations for all users */
    val itemFactors = model.productFeatures.map { case (id, factor) => factor }.collect()
    val itemMatrix = new DoubleMatrix(itemFactors)
    println(itemMatrix.rows, itemMatrix.columns)
    // (1682,50)

    // broadcast the item factor matrix
    val imBroadcast = sc.broadcast(itemMatrix)

    // compute recommendations for each user, and sort them in order of score so that the actual input
    // for the APK computation will be correct
    val allRecs = model.userFeatures.map{ case (userId, array) =>
      val userVector = new DoubleMatrix(array)
      val scores = imBroadcast.value.mmul(userVector)
      val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
      val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
      (userId, recommendedIds)
    }

    // next get all the movie ids per user, grouped by user id
    val userMovies = ratings.map{ case Rating(user, product, rating) => (user, product) }.groupBy(_._1)
    // userMovies: org.apache.spark.rdd.RDD[(Int, Seq[(Int, Int)])] = MapPartitionsRDD[277] at groupBy at <console>:21

    // finally, compute the APK for each user, and average them to find MAPK
    val K = 10
    val MAPK = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
      avgPrecisionK(actual, predicted, K)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision at K = " + MAPK)
    // Mean Average Precision at K = 0.030486963254725705

    /* Using MLlib built-in metrics */

    // MSE, RMSE and MAE
    import org.apache.spark.mllib.evaluation.RegressionMetrics
    val predictedAndTrue = ratingsAndPredictions.map { case ((user, product), (actual, predicted)) => (actual, predicted) }
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)
    println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)
    // Mean Squared Error = 0.08231947642632852
    // Root Mean Squared Error = 0.2869137090247319

    // MAPK
    import org.apache.spark.mllib.evaluation.RankingMetrics
    val predictedAndTrueForRanking = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2)
      (predicted.toArray, actual.toArray)
    }
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
    println("Mean Average Precision = " + rankingMetrics.meanAveragePrecision)
    // Mean Average Precision = 0.07171412913757183

    // Compare to our implementation, using K = 2000 to approximate the overall MAP
    val MAPK2000 = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
      avgPrecisionK(actual, predicted, 2000)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision = " + MAPK2000)
    // Mean Average Precision = 0.07171412913757186


  }

}

