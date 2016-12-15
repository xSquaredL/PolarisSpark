import scala.io.Source
import java.net.{URL, InetSocketAddress}
import java.net.Proxy
import javax.net.ssl.HttpsURLConnection;
import java.net.Authenticator
import java.net.PasswordAuthentication
import util.control.Breaks._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer, StopWordsRemover,VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.sql.functions.{concat, lit}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.clustering.LDA

object FakeReviewDetector {
  val conf = new SparkConf()
        .setAppName("PolarisSparkApp")
  val sc = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  import sqlContext.implicits._

  val clean_html_pattern = "(\\n|\")".r

  @throws(classOf[java.io.IOException])
  @throws(classOf[java.net.SocketTimeoutException])
  def get(url: String,
        connectTimeout: Int = 5000,
        readTimeout: Int = 5000,
        requestMethod: String = "GET") =
  {
    System.setProperty("http.proxyHost", "proxy.troweprice.com");
    System.setProperty("http.proxyPort", "8080");
    System.setProperty("https.proxyHost", "proxy.troweprice.com");
    System.setProperty("https.proxyPort", "8080");

    val connection = (new URL(url)).openConnection.asInstanceOf[HttpsURLConnection]
    connection.setConnectTimeout(connectTimeout)
    connection.setReadTimeout(readTimeout)
    connection.setRequestMethod(requestMethod)
    val inputStream = connection.getInputStream
    val content = Source.fromInputStream(inputStream).mkString
    if (inputStream != null) inputStream.close
    Thread.sleep(200)
    content
  }

  def parse_business_page (html:String) : ArrayBuffer[Tuple7[Int, String, Double, String, Int, Int, String]] = {
    val buf = scala.collection.mutable.ArrayBuffer.empty[Tuple7[Int, String, Double, String, Int, Int, String]]
    val cleaned_html = clean_html_pattern.replaceAllIn(html, "").toLowerCase()

    val review_pattern = """<div id=review_(\d+).+?profile_(\w+).+?ui_bubble_rating bubble_(\d).+?<div class=entry><p class=partial_entry>(.+?)<\/p>""".r
    val date_pattern = """ratingdate relativedate title=(\w+)\s(\d+),\s(\d+)>""".r
    review_pattern.findAllIn(cleaned_html).matchData foreach {
      m =>
        val review_id = m.group(1).toInt
        val uid = m.group(2)
        val rating = m.group(3).toDouble
        var month = ""
        var day = 0
        var year = 0
        var review_text = m.group(4)
        if(review_text.endsWith("...")){
          review_text = ""
        }
        date_pattern.findAllIn(m.toString()).matchData foreach {
          n =>
              month = n.group(1)
              day = n.group(2).toInt
              year = n.group(3).toInt
        }
        if(month.isEmpty){
            val date_pattern_old = """ratingdate.+?reviewed (\w+)\s(\d+),\s(\d+)""".r
            date_pattern_old.findAllIn(m.toString()).matchData foreach {
              n =>
                  month = n.group(1)
                  day = n.group(2).toInt
                  year = n.group(3).toInt
            }
        }
        val review:Tuple7[Int, String, Double, String, Int, Int, String] = (review_id, uid, rating, month, day, year, review_text)
        buf += review
    }
    val size = review_pattern.findAllIn(cleaned_html).matchData.size
    println(size)
    return buf
  }

  def fetch_reviews (offset:Int, business_base_url: String) : ArrayBuffer[Tuple7[Int, String, Double, String, Int, Int, String]] = {
    var reviews_url = s"$business_base_url-or$offset"
    val page = get(reviews_url)

    return parse_business_page(page)
  }

  def fetch_review_by_id (location:String, business:String, row:Tuple7[Int, String, Double, String, Int, Int, String]) : Tuple7[Int, String, Double, String, Int, Int, String] = {
    val id = row._1
    var text = row._7
    if(text.isEmpty){
      var review_url = s"https://www.tripadvisor.com/ShowUserReviews-g$location-d$business-r$id"
      val page = get(review_url)
      val cleaned_html = clean_html_pattern.replaceAllIn(page, "").toLowerCase()

      val review_pattern = s"id=review_$id>(.*?)<\\/p>".r
      review_pattern.findAllIn(cleaned_html).matchData foreach {
        m => text = m.group(1)
      }
      return (id, row._2, row._3, row._4, row._5, row._6, text)
    } else {
      return row
    }

  }

  def fetch_user (row:Tuple7[Int, String, Double, String, Int, Int, String]) : Seq[Any] = {
    var user_info = new ArrayBuffer[Any]()

    val uid = row._2.toUpperCase()
    var user_url = s"https://www.tripadvisor.com/MemberOverlay?uid=$uid"
    val page = get(user_url)

    val cleaned_html = clean_html_pattern.replaceAllIn(page, "").toLowerCase()
    val user_pattern = "href=\\/members\\/([\\w-_]+?)>".r
    val member_since_pattern = "tripadvisor member since (\\d{4})".r
    val level_pattern = "level.+?(\\d).+?contributor".r
    val excellent_pattern = "excellent.+?\\((\\d+)\\)".r
    val very_good_pattern = "very\\sgood.+?\\((\\d+)\\)".r
    val average_pattern = "average.+?\\((\\d+)\\)".r
    val poor_pattern = "poor.+?\\((\\d+)\\)".r
    val terrible_pattern = "terrible.+?\\((\\d+)\\)".r
    user_pattern.findFirstMatchIn(cleaned_html) match {
      case Some(m) => user_info += m.group(1)
      case None => user_info += ""
    }
    member_since_pattern.findFirstMatchIn(cleaned_html) match {
      case Some(m) => user_info += (2016 - m.group(1).toDouble)
      case None => user_info += 0.0
    }
    level_pattern.findFirstMatchIn(cleaned_html) match {
      case Some(m) => user_info += m.group(1).toDouble
      case None => user_info += 0.0
    }
    excellent_pattern.findFirstMatchIn(cleaned_html) match {
      case Some(m) => user_info += m.group(1).toDouble
      case None => user_info += 0.0
    }
    very_good_pattern.findFirstMatchIn(cleaned_html) match {
      case Some(m) => user_info += m.group(1).toDouble
      case None => user_info += 0.0
    }
    average_pattern.findFirstMatchIn(cleaned_html) match {
      case Some(m) => user_info += m.group(1).toDouble
      case None => user_info += 0.0
    }
    poor_pattern.findFirstMatchIn(cleaned_html) match {
      case Some(m) => user_info += m.group(1).toDouble
      case None => user_info += 0.0
    }
    terrible_pattern.findFirstMatchIn(cleaned_html) match {
      case Some(m) => user_info += m.group(1).toDouble
      case None => user_info += 0.0
    }
    val row_seq = Seq(row._1,row._2,row._3,row._4,row._5,row._6,row._7) ++ Seq(user_info(0),user_info(1),user_info(2),user_info(3),user_info(4),user_info(5),user_info(6),user_info(7))
    return row_seq
  }

  def main(args: Array[String]) {
  val location = args(0)
  val business = args(1)
  val request = args(2)

  var business_base_url = s"https://www.tripadvisor.com/Hotel_Review-g$location-d$business"


    val page1 = get(business_base_url)
    // get total page numbers
    var last_page_offset = 0

    val offset_pattern = "data-offset=\"(\\d+)\"".r
    offset_pattern.findAllIn(page1).matchData foreach {
      m => last_page_offset = m.group(1).toInt
    }

    val page_1_reviews = parse_business_page(page1)

    val offset_seq = 10 to last_page_offset by 10
    val offset_rdd = sc.parallelize(offset_seq)

    val downloaded = offset_rdd.flatMap(x => fetch_reviews(x, business_base_url))
    val page_1_review_rdd = sc.parallelize(page_1_reviews.toSeq)
    val all_reviews_rdd = downloaded.union(page_1_review_rdd)

    val all_reviews_filled_rdd = all_reviews_rdd.map(x => fetch_review_by_id(location, business, x))

    val all_reviews_with_user_rdd = all_reviews_filled_rdd.map(x => fetch_user(x)).map(x => (x(0).asInstanceOf[Int],x(1).asInstanceOf[String],x(2).asInstanceOf[Double],x(3).asInstanceOf[String],x(4).asInstanceOf[Int],x(5).asInstanceOf[Int],x(6).asInstanceOf[String],x(7).asInstanceOf[String],x(8).asInstanceOf[Double],x(9).asInstanceOf[Double],x(10).asInstanceOf[Double],x(11).asInstanceOf[Double],x(12).asInstanceOf[Double],x(13).asInstanceOf[Double],x(14).asInstanceOf[Double]))

    val df = all_reviews_with_user_rdd.toDF("id", "uid", "rating","month","day","year","review","user_id","member_year","level","excellent","very_good","average","poor","terrible")

    val training_df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("delimiter", "\t")
    .load("hwasbs://polaris-spark-training@polaris598.blob.core.windows.net/*")
    .withColumn("isTraining", lit(true))

    val prediction_df = df.withColumn("isTraining", lit(false))
    val df_union = prediction_df.unionAll(training_df)

    val tokenizer = new RegexTokenizer().setInputCol("review").setOutputCol("words").setPattern("[a-z']+").setGaps(false)
    val stopWordsRemover = new StopWordsRemover()
          .setInputCol("words")
          .setOutputCol("tokens")

    val countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    val assembler = new VectorAssembler()
      .setInputCols(Array("features", "rating","member_year","excellent","very_good","average","poor","terrible"))
      .setOutputCol("featureset")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, countVectorizer, idf, assembler))

    val model1 = pipeline.fit(df_union)
    val model2 = model1.transform(df_union)


    val toCSV = udf((vec: SparseVector) => vec.toArray.mkString("\t"))

    val new_df = model2.withColumn("features_csv", toCSV($"featureset"))
    new_df.filter($"isTraining"===true).select(concat($"id", lit("\t"), $"features_csv")).repartition(1).write.text(s"wasbs://polaris-spark-out@polaris598.blob.core.windows.net/$request-training")
    new_df.filter($"isTraining"===false).select(concat($"id", lit("\t"), $"features_csv")).repartition(1).write.text(s"wasbs://polaris-spark-out@polaris598.blob.core.windows.net/$request-test")
    new_df.filter($"isTraining"===false).select(concat($"id", lit("\t"), $"review")).repartition(1).write.text(s"wasbs://polaris-spark-out@polaris598.blob.core.windows.net/$request-reviews")

    var lda_dataset = model2.select("features")
        val lda = new LDA().setK(5).setMaxIter(10)
        val ldaModel = lda.fit(lda_dataset)
        val vocabArray = model1.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
        val topicIndices = ldaModel.describeTopics(20)
        val topics = scala.collection.mutable.ListBuffer.empty[String]
        topicIndices.collect().foreach(r => {
          val arr: Seq[Integer] = r.getAs[Seq[Integer]](1)
          arr.foreach { e =>
                 val v = vocabArray(e)
                 topics += v
          }
        })
        val topic_df = sqlContext.sparkContext.parallelize(topics.toSeq).toDF()
    topic_df.repartition(1).write.text(s"wasbs://polaris-spark-out@polaris598.blob.core.windows.net/$request-lda")

    }
}
