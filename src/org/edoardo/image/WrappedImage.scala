package org.edoardo.image

import ij.process.ImageProcessor
import ij.{IJ, ImagePlus}
import net.imglib2.FinalInterval
import net.imglib2.`type`.numeric.integer.UnsignedByteType
import net.imglib2.algorithm.gradient.PartialDerivative
import net.imglib2.img.Img
import net.imglib2.img.array.ArrayImgFactory
import net.imglib2.img.display.imagej.ImageJFunctions
import net.imglib2.util.Intervals
import net.imglib2.view.Views
import org.edoardo.segmentation.SegmentationResult

import scala.Array.ofDim

class WrappedImage(val originalImage: ImagePlus, val windowing: (Int, Int) = (0, 0)) {
	def getGradient(x: Int, y: Int, z: Int): Int = {
		(0 until dimensions).map(dir => gradientProcessors(dir)(z).getPixel(x, y)).max
	}
	
	val width: Int = originalImage.getWidth
	val height: Int = originalImage.getHeight
	val depth: Int = originalImage.getDimensions()(3)
	
	val image: ImagePlus = IJ.createImage("windowed", "8-bit", width, height, depth)
	
	for (z <- 1 to depth) {
		image.setZ(z)
		originalImage.setZ(z)
		for (x <- 0 until width; y <- 0 until height) {
			if (windowing != (0, 0))
				image.getProcessor.putPixel(x, y,
					127 + Math.round(255 * ((originalImage.getPixel(x, y)(0) - windowing._1).toFloat / windowing._2)))
			else
				image.getProcessor.putPixel(x, y, originalImage.getPixel(x, y)(0))
		}
	}
	
	val wrapped: Img[UnsignedByteType] = ImageJFunctions.wrapByte(image)
	
	val dimensions: Int = if (depth == 1) 2 else 3
	val processors: Array[ImageProcessor] = (0 until depth).map(z => image.getImageStack.getProcessor(z + 1)).toArray
	var gradientProcessors: Array[Array[ImageProcessor]] = _
	
	def getVoxel(x: Int, y: Int, z: Int): Int = processors(z).getPixel(x, y)
	
	def doPreProcess(): Unit = {
		computeGradientImage()
	}
	
	def contains(x: Int, y: Int): Boolean = x >= 0 && y >= 0 && x < width && y < height
	
	def toSegmentationResult(stayInLayer: Int): SegmentationResult = {
		if (stayInLayer == -1) {
			val result: Array[Array[Array[Boolean]]] = ofDim[Boolean](width, height, depth)
			for (x <- 0 until width; y <- 0 until height; z <- 0 until depth)
				result(x)(y)(z) = getVoxel(x, y, z).equals(255)
			new SegmentationResult(result)
		} else {
			val result: Array[Array[Array[Boolean]]] = ofDim[Boolean](width, height, 1)
			for (x <- 0 until width; y <- 0 until height)
				result(x)(y)(0) = getVoxel(x, y, stayInLayer).equals(255)
			new SegmentationResult(result)
		}
	}
	
	private def computeGradientImage(): Unit = {
		val gradients: Img[UnsignedByteType] = new ArrayImgFactory[UnsignedByteType]().create(
			if (dimensions == 2) Array(width, height, 2)
			else Array(width, height, depth, dimensions), new UnsignedByteType())
		val gradientComputationInterval: FinalInterval = Intervals.expand(wrapped, -1)
		for (d <- 0 until dimensions)
			PartialDerivative.gradientCentralDifference(wrapped,
				Views.interval(Views.hyperSlice(gradients, dimensions, d), gradientComputationInterval), d)
		val gradientImage: ImagePlus = ImageJFunctions.wrap(gradients, image.getShortTitle + "-gradients")
		gradientProcessors = (0 until dimensions).map(dir => {
			(0 until depth).map(z => gradientImage.getImageStack.getProcessor((dir * depth) + z + 1)).toArray
		}).toArray
	}
}