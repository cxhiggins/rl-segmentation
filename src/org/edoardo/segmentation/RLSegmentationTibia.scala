package org.edoardo.segmentation

import java.io.File

import ij.IJ
import ij.io.{FileSaver, Opener}
import ij.plugin.FolderOpener
import org.edoardo.image.{Raw, WindowedImage}
import org.edoardo.parser.{IPF, MFS, VolumeIPF}
import org.edoardo.rl.Policy

import scala.collection.mutable

/**
  * Contains the main code for running the segmentation algorithm.
  */
object RLTibiaSegmentation {
	val policy = new Policy[Decision, RegionInfo]
	val opener = new Opener()
	val regionInfoCache: mutable.Map[Int, RegionInfo] = mutable.Map.empty
	var epsilonReciprocal = 10
	val dataLocation = "/fs/medimaging/datasource/Unicompartmental_knee/UnicompartmentalKneeSetStephenMellon/"
	val localDataLocation = "/Users/cara/Desktop/oxford/3rd Year Project/datasource/Unicompartmental_knee/"
	
	/**
	  * Main method to run our segmentation algorithm.
	  * @param args A number between one and three containing the number of the experiment to run.
	  */
	def main(args: Array[String]): Unit = {
		if (args(0) == "1")
			experiementOne()
		else
			println("Invalid experiment number.")
	}
	
	/**
	  * The first experiment we ran on MRI scans from the Botnar Research Centre.
	  */
	def experiementOne(): Unit = {
		val imageInfos = List(
			// Training data
			// ImageInfo(id: Integer, fileName: String, layer: Integer, seed: (Int, Int, Int), windowing: (Int, Int))
			// Using default layer 1 for now, with default windowing
			ImageInfo(116, dataLocation+"116.jpg", -1, (375, 360, 0), (127, 256)),
			ImageInfo(173, dataLocation+"173ap.jpg", -1, (165, 450, 0), (127, 256)),
			ImageInfo(226, dataLocation+"226ap_1_1.jpg", -1, (222, 305, 0), (127, 256)),
			ImageInfo(239, dataLocation+"239ap.jpg", -1, (440, 642, 0), (127, 256)),
			ImageInfo(244, dataLocation+"244ap.jpg", -1, (240, 305, 0), (127, 256)),
			
			// Evaluation data
			// ImageInfo(245, dataLocation+"245ap.jpg", -1, (263, 315, 0), (127, 256)),
			// ImageInfo(250, dataLocation+"250ap.jpg", -1, (510, 580, 0), (130, 120))
			// ImageInfo(251, dataLocation+"251ap.jpg", -1, (234, 286, 0), (127, 160))
			// ImageInfo(253, dataLocation+"253ap.jpg", -1, (250, 330, 0), (127, 256)),
			ImageInfo(260, dataLocation+"260ap.jpg", -1, (223, 295, 0), (170, 110))
		)

		val resultsFolder = "experimentOneResults/"

		println("-- Before Training --")
		for (imageInfo <- imageInfos)
			doImage(imageInfo.fileName, imageInfo.fileName.dropRight(3) + "ipf",
				resultsFolder+"preTraining-" + imageInfo.id + ".tiff", imageInfo.seed, imageInfo.windowing,
				Some(imageInfo.fileName.dropRight(4) + "-tibiaGT.png"), imageInfo.layer, 0, saveAsRaw = false)
		
		println("-- Training --")
		for (imageInfo <- imageInfos.take(5))
			doImage(imageInfo.fileName, imageInfo.fileName.dropRight(3) + "ipf",
				resultsFolder+"training-" + imageInfo.id + ".tiff", imageInfo.seed, imageInfo.windowing,
				Some(imageInfo.fileName.dropRight(4) + "-tibiaGT.png"), imageInfo.layer, 40, saveAsRaw = false)
		
		printPolicy()
		
		println("-- After Training --")
		for (imageInfo <- imageInfos)
			doImage(imageInfo.fileName, imageInfo.fileName.dropRight(3) + "ipf",
				resultsFolder+"postTraining-" + imageInfo.id + ".tiff", imageInfo.seed, imageInfo.windowing,
				Some(imageInfo.fileName.dropRight(4) + "-tibiaGT.png"), imageInfo.layer, 0, saveAsRaw = false)
	}
	
	/**
	  * Apply our algorithm to an image.
	  *
	  * @param name            the name of the file (or folder) the image (or layers of the image) can be found in
	  * @param ipfName         the name of the file containing the IPF for the image
	  * @param resultName      the name of the result file to store the segmentation result in, should end in .tiff
	  * @param seed            the seed point to begin growing the region from
	  * @param windowing       the windowing to use, in the form of a pair of (centre, width)
	  * @param gtName          the name of the file containing the gold standard to compare with (and learn from, if applicable)
	  * @param stayInLayer     the layer to explore in (-1 to explore the whole image)
	  * @param numPracticeRuns the number of times to practice on this image (0 to not train on this image)
	  * @param saveAsRaw       whether to save the image as RAW (will be saved as TIFF otherwise)
	  */
	def doImage(name: String, ipfName: String, resultName: String, seed: (Int, Int, Int), windowing: (Int, Int) = (0, 0),
				gtName: Option[String] = None, stayInLayer: Integer = -1, numPracticeRuns: Int = 40, saveAsRaw: Boolean): Unit = {
		val img: WindowedImage = new WindowedImage(
			if (name.takeRight(3) == "mhd")
				Raw.openMetadata(name, stayInLayer + 1)
			else if (new File(name).isDirectory)
				new FolderOpener().openFolder(name)
			else IJ.openImage(name), windowing)
		new FileSaver(img.image).saveAsTiff(resultName + "-original.tiff")
		val ipf: VolumeIPF = IPF.loadFromFile(ipfName)
		val gt: Option[SegmentationResult] = gtName.map(name =>
			if (name.takeRight(3) == "mfs")
				MFS.loadFromFile(name, ipf)
			else
				new WindowedImage(
					if (name.takeRight(3) == "mhd")
						Raw.openLabels(name)
					else opener.openImage(name)
				).toSegmentationResult(stayInLayer))
		if (gt.isDefined)
			gt.get.writeTo(resultName + "-gt", saveAsRaw)
		img.doPreProcess()
		if (gt.isDefined) {
			for (i <- 0 until numPracticeRuns) {
				val result: SegmentationResult = analyseImage(img, ipf, gt, seed, stayInLayer != -1)
				println(name + "\t" + i + "\t" + score(result, gt.get))
				result.writeTo(resultName + "-" + i, saveAsRaw)
			}
		}
		val result: SegmentationResult = analyseImage(img, ipf, None, seed, stayInLayer != -1)
		if (gt.isDefined)
			println(name + "\tfin\t" + score(result, gt.get))
		result.writeTo(resultName, saveAsRaw)
		regionInfoCache.clear()
	}
	
	/**
	  * Get the information for a given region from the first branch layer of the IPF.
	  *
	  * @param region the identifier for the region
	  * @param ipf    the IPF of the image we are considering
	  * @param img    the image we are considering
	  * @return the information to be used by the agent to decide whether or not to include this region
	  */
	def getInfo(region: Int, ipf: VolumeIPF, img: WindowedImage): RegionInfo = {
		regionInfoCache.getOrElseUpdate(region, {
			val pixels: List[(Int, Int, Int)] = ipf.getRegionPixels(region)
			val avgIntensity: Int = pixels.map(p => img.getVoxel(p._1, p._2, p._3)).sum / pixels.size
			// val minIntensity: Int = pixels.map(p => img.getVoxel(p._1, p._2, p._3)).min
			// val maxIntensity: Int = pixels.map(p => img.getVoxel(p._1, p._2, p._3)).max
			// val avgGradient: Int = pixels.map(p => img.getGradient(p._1, p._2, p._3)).sum / pixels.size
			val maxGradient: Int = pixels.map(p => img.getGradient(p._1, p._2, p._3)).max
			RegionInfo(List(avgIntensity, maxGradient))
		})
	}
	
	/**
	  * Analyse the given image.
	  *
	  * @param img         the image to analyse
	  * @param ipf         the IPF for the image
	  * @param gt          the gold standard to compare to, if applicable
	  * @param seed        the seed point to grow from
	  * @param stayInLayer whether or not to remain in the same layer
	  * @return the result of segmenting the image
	  */
	def analyseImage(img: WindowedImage, ipf: VolumeIPF, gt: Option[SegmentationResult], seed: (Int, Int, Int),
					 stayInLayer: Boolean): SegmentationResult = {
		if (gt.isDefined)
			assert(img.width == gt.get.width && img.height == gt.get.height && img.depth == gt.get.depth)
		val selection = new Selection(img.height, img.width, img.depth, ipf, stayInLayer)
		var decisions: List[(RegionInfo, Int, Decision)] = List()
		selection.startPixel(seed._1, seed._2, seed._3, if (gt.isDefined) 2 else 3)
		while (!selection.completed()) {
			val region: Int = selection.getRegion
			val state: RegionInfo = getInfo(region, ipf, img)
			val decision: Decision =
				if (gt.isEmpty) policy.greedyPlay(state)
				else policy.epsilonSoft(state, epsilonReciprocal)
			decisions ::= (state, region, decision)
			if (decision.include)
				selection.includeRegion(region)
			else
				selection.excludeRegion(region)
		}
		if (gt.isDefined)
			decisions.foreach {
				case (state, region, dec) =>
					policy.update(state, dec, reward(region, dec.include, ipf, gt))
			}
		val result: SegmentationResult = selection.getResult
		result.closeResult()
		result
	}
	
	/**
	  * Calculates the reward to give our agent for deciding to include or exclude a given region.
	  *
	  * @param region   the region considered
	  * @param decision whether or not the agent chose to include it
	  * @param ipf      the IPF for the
	  * @param gt       the gold standard we are comparing against (this function will return constant 0 if this is None)
	  * @return a value corresponding to how many more pixels the decision was correct for (so, this will be a positive
	  *         value if the correct decision was made, and negative otherwise)
	  */
	def reward(region: Int, decision: Boolean, ipf: VolumeIPF, gt: Option[SegmentationResult]): Int = {
		if (gt.isEmpty) return 0
		val pixels: List[(Int, Int, Int)] = ipf.getRegionPixels(region)
		var reward: Int = 0
		for ((x, y, z) <- pixels)
			reward += (if (gt.get.doesContain(x, y, z)) 1 else -1)
		if (decision) reward
		else -reward
	}
	
	/**
	  * Create a string containing the scores of a segmentation compared to a gold standard.
	  *
	  * @param result the result of a segmentation
	  * @param gt     the gold standard we are comparing against
	  * @return A tab separated String of values consisting of the DSC, TPVF and FPVF of the segmentation.
	  */
	def score(result: SegmentationResult, gt: SegmentationResult): String = {
		var overlap = 0
		var resultSize = 0
		var gtSize = 0
		var falsePositive = 0
		val imageSize: Int = gt.height * gt.width * gt.depth
		for (x <- 0 until result.width; y <- 0 until result.height; z <- 0 until result.depth) {
			if (result.doesContain(x, y, z) && gt.doesContain(x, y, z)) overlap += 1
			if (result.doesContain(x, y, z)) resultSize += 1
			if (gt.doesContain(x, y, z)) gtSize += 1
			if (result.doesContain(x, y, z) && !gt.doesContain(x, y, z)) falsePositive += 1
		}
		(2f * overlap) / (gtSize + resultSize) + "\t" +
			overlap.toFloat / gtSize + "\t" +
			falsePositive.toFloat / (imageSize - gtSize)
	}
	
	/**
	  * Prints out the current policy (ie. what the agent believes to be the best action in every state).
	  */
	def printPolicy(): Unit = {
		println("-- Policy Learnt --")
		for (x <- 0 to 255) {
			for (y <- 0 to 255)
				print((if (!policy.haveEncountered(RegionInfo(List(x, y)))) 0
				else if (policy.greedyPlay(RegionInfo(List(x, y))).include) 1
				else -1) + "\t")
			println()
		}
		printPercentageSeen()
	}
	
	/**
	  * Prints out the percentage of theoretically possible states actually encountered.
	  */
	def printPercentageSeen(): Unit = {
		var total = 0
		var seen = 0
		for (x <- 0 to 255; y <- 0 to 255) {
			total += 1
			if (policy.haveEncountered(RegionInfo(List(x, y))))
				seen += 1
		}
		println("Percentage of states seen: " + 100 * (seen.toFloat / total))
	}
	
	private case class ImageInfo(id: Integer, fileName: String, layer: Integer, seed: (Int, Int, Int), windowing: (Int, Int))
	
	private case class XRayInfo(id: Integer, gt: String, seed: (Int, Int, Int))
	
}