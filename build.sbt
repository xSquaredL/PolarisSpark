name := "PolarisSparkApp"
version := "1.0"
scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "1.6.2",
	"org.apache.spark" %% "spark-sql" % "1.6.2",
	"org.apache.spark" %% "spark-mllib" % "1.6.2",
	"org.apache.commons" % "commons-lang3" % "3.5"
)


