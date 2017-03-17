package org.edoardo.segmentation

import org.edoardo.bitmap.WrappedImage
import org.edoardo.parser.VolumeIPF

import scala.collection.mutable

class Selection(val height: Int, width: Int, depth: Int, ipf: VolumeIPF, stayInLayer: Boolean) {
	val toConsider: mutable.Set[Int] = mutable.Set[Int]()
	var toConsiderQueue: mutable.Queue[Int] = mutable.Queue[Int]()
	val excluded: mutable.Set[Int] = mutable.Set[Int]()
	val included: mutable.Set[Int] = mutable.Set[Int]()
	var firstZ: Int = -1
	
	def completed(): Boolean = toConsider.isEmpty
	
	def getRegion: Int = {
		val result: Int = toConsiderQueue.dequeue()
		toConsider.remove(result)
		result
	}
	
	def includeRegion(region: Int): Unit = {
		included += region
		for (neighbour <- ipf.getNeighbours(region)) {
			if (!excluded.contains(neighbour) && !included.contains(neighbour)) {
				if (!(stayInLayer && ipf.getZ(region) != firstZ) && toConsider.add(neighbour))
					toConsiderQueue.enqueue(neighbour)
			}
		}
	}
	
	def startPixel(x: Int, y: Int, z: Int, img: WrappedImage, layer: Int): Unit = {
		val startRegions: List[Int] = ipf.getRegionsInLayer(layer, x, y, z)
		firstZ = z
		for (region <- startRegions)
			includeRegion(region)
	}
	
	def excludeRegion(region: Int): Unit = {
		excluded += region
	}
	
	def getResult: SegmentationResult = {
		val status: Array[Array[Array[Boolean]]] = Array.ofDim(width, height, depth)
		for ((x, y, z) <- included.flatMap(region => ipf.getRegionPixels(region)))
			status(x)(y)(z) = true
		new SegmentationResult(status)
	}
	
}
