package org.edoardo.segmentation

import java.io.File

import ij.IJ
import ij.io.{FileSaver, Opener}
import ij.plugin.FolderOpener
import org.edoardo.image.{Raw, WindowedImage}
import org.edoardo.parser.{IPF, MFS, VolumeIPF, Node, BranchLayer}

import scala.collection.mutable

/**
  * Describes a component of the IPF, composed of adjacent nodes and the component's associated properties.
  */
case class Component(var nodes: mutable.Set[Int], var voxelCount: Int)

/**
  * Contains the main code for running the segmentation algorithm.
  */
object UCKneeSegmentation {
	val opener = new Opener()
	var epsilonReciprocal = 10
    var windMinX, windMinY, windMaxX, windMaxY = 0
	
	/**
	  * Main method to run our segmentation algorithm.
	  * @param args A number between one and three containing the number of the experiment to run.
	  */
	def main(args: Array[String]): Unit = {
		if (args(0) == "uc")
			ucKneeExperimentOne()
		else
			println("Invalid experiment number.")
	}

	def ucKneeExperimentOne(): Unit = {
		val imageInfos = List(
			// Training data
			// ImageInfo(id: Integer, fileName: String, layer: Integer, seed: (Int, Int, Int), windowing: (Int, Int))
			ImageInfo(1, "/Users/cara/Desktop/oxford/3rd Year Project/datasource/Unicompartmental_knee/116.jpg", 1, (200, 200, 0), (0, 400))
		)

        val layerIndex = 1

		for (imageInfo <- imageInfos)
			segmentUCKnee(imageInfo.fileName, imageInfo.fileName.substring(0, imageInfo.fileName.length - 4) + ".ipf",
				"preTraining-" + imageInfo.id, imageInfo.seed, imageInfo.windowing,
				None, imageInfo.layer, 0, saveAsRaw = false, layerIndex)
	}

	/**
	  * Apply the UC knee segmentation algorithm to an image.
	  *
	  * @param name            the name of the file (or folder) the image (or layers of the image) can be found in
	  * @param ipfName         the name of the file containing the IPF for the image
	  * @param resultName      the name of the result file to store the segmentation result in, should end in .tiff
	  * @param seed            the seed poInt to begin growing the region from
	  * @param windowing       the windowing to use, in the form of a pair of (centre, width)
	  * @param gtName          the name of the file containing the gold standard to compare with (and learn from, if applicable)
	  * @param stayInLayer     the layer to explore in (-1 to explore the whole image)
	  * @param numPracticeRuns the number of times to practice on this image (0 to not train on this image)
	  * @param saveAsRaw       whether to save the image as RAW (will be saved as TIFF otherwise)
	  */
	def segmentUCKnee(name: String, ipfName: String, resultName: String, seed: (Int, Int, Int), windowing: (Int, Int) = (0, 0),
				gtName: Option[String] = None, stayInLayer: Integer = -1, numPracticeRuns: Int = 40, saveAsRaw: Boolean, layerIndex: Int): Unit = {
		val img: WindowedImage = new WindowedImage(IJ.openImage(name), windowing);

		new FileSaver(img.image).saveAsTiff(resultName + "-original.tiff")
		val ipf: VolumeIPF = IPF.loadFromFile(ipfName)
		val result: mutable.Set[Int] = selectKnee(img, ipf, None, layerIndex)
		// result.writeTo(resultName, saveAsRaw)
	}

    /**
      * @brief	Select region of interest (ROI) by setting window minima and maxima
      * 
      * @param	volumeSize	    [x width, y height, z count]
      * @param  distFromEdge    User-selected parameter indicating the minimum distance a node must be from the edge; 
                                    prevents selection of border nodes, which often have very bright values
      *
      * @return a tuple containing the minimum x and y coordinates and the maximum of the selected ROI
      */
    def selectROI(ipf: VolumeIPF, distFromEdge: Int): (Int, Int, Int, Int) = {
        assert(distFromEdge <= ipf.width / 4 && distFromEdge <= ipf.height / 4)
        val htow = ipf.height * 1.0 / ipf.width;

        var windMinY = distFromEdge;
        var windMinX = distFromEdge;
        var windMaxX = ipf.width-distFromEdge;
        var windMaxY = ipf.height-distFromEdge;

        // println("htow: "+htow)

        // Determine region of focus based on image size
        if (0.75 < htow && htow < 1.5) { // Roughly square
            windMinX = ipf.width / 4;
            windMaxX = ipf.width * 3 / 4;

            windMinY = ipf.height / 4;
            windMaxY = ipf.height * 3 / 4;
        } else if (htow > 1.5) {
            windMinY = ipf.height / 4;
            windMaxY = ipf.height * 3 / 4;
        } else {
            windMinX = ipf.width / 4;
            windMaxX = ipf.width * 3 / 4;
        }

        return (windMinX, windMinY, windMaxX, windMaxY)
    }
	
	/**
	  * Analyse the given image.
	  *
	  * @param img         the image to analyse
	  * @param ipf         the IPF for the image
	  * @param gt          the gold standard to compare to, if applicable
	  * @return the result of segmenting the UC knee prosthesis
	  */
	def selectKnee(img: WindowedImage, ipf: VolumeIPF, gt: Option[SegmentationResult], layerIndex: Int): mutable.Set[Int] = {
        require(0 <= layerIndex && layerIndex <= ipf.branchLayers.length, "layerIndex out of bounds.")
         
        val layer = ipf.branchLayers(ipf.branchLayers.length - layerIndex);

        if (windMinX == windMaxX) { // Only call selection method if ROI is not already initialized
            val roi = selectROI(ipf, 10)
            windMinX = roi._1
            windMinY = roi._2
            windMaxX = roi._3
            windMaxY = roi._4
        }

        // println("windMin: ("+windMinX+", "+windMinY+") -> windMax: ("+windMaxX+", "+windMaxY+")")
        
        var (nodes, maxMeanGrey): (mutable.ListBuffer[Int], Float) = max_mean_grey_with_filter(
            layer, windMinX, windMinY, windMaxX, windMaxY
        );
        
        val threshold = maxMeanGrey - 20;
        val connectedComps: mutable.ListBuffer[Component] = connected_components(nodes, ipf, layer, threshold)
        connectedComps.sortBy(comp => comp.voxelCount)    // Sort connected components by voxel count

        // Initialize region to store unicompartmental knee node indices
        // PartitionForestSelection_Ptr filledRegion(new PartitionForestSelectionT(volume_ipf()));
        var uckneeNodeIndices = mutable.Set[Int]()

        // Select the two largest components and set the ucknee variable accordingly
        for (i <- 0 until 2.min(connectedComps.length)) {
            val comp = connectedComps(i);
            uckneeNodeIndices ++= comp.nodes
        }

        // println("Selected top 2 connected components")
        println("UC knee: "+uckneeNodeIndices)

        return uckneeNodeIndices
    }

    def max_mean_grey_with_filter(layer: BranchLayer, windMinX: Int, windMinY: Int, windMaxX: Int, windMaxY: Int): (mutable.ListBuffer[Int], Float) = {  
        var maxMeanGrey = 0.toFloat;
        val nodes = new mutable.ListBuffer[Int]();
        
        for ((nodeI, node) <- layer.nodes) {
            if (
                node.voxelCount <= 15000 && node.voxelCount >= 10 && // Node must contain between 100 and 15,000 voxels
                windMinX < node.xMin && node.xMax < windMaxX && windMinY < node.yMin && node.yMax < windMaxY // Region lies within region of interest (not touching edges)
            ) { 
                nodes += nodeI;  // Add node index to results list
                if (node.meanGrey > maxMeanGrey)
                    maxMeanGrey = node.meanGrey;
            }
        }

        return (nodes, maxMeanGrey);
    }

    def connected_components(nodes: mutable.ListBuffer[Int], ipf: VolumeIPF, layer: BranchLayer, threshold: Float): mutable.ListBuffer[Component] = {
        var connectedComps = mutable.ListBuffer[Component]()

        for (nodeI <- nodes) {
            val node = layer.nodes(nodeI)
            if (node.meanGrey > threshold) { // Check that mean grey value is above the specified threshold 
                var nodeAddedTo = -1

                // Obtain set of adjacent nodes
                val adjNodeSet = ipf.getNeighbours(nodeI).toSet

                // Iterate through connected components
                var compI = 0
                while (compI < connectedComps.length) {
                    val component = connectedComps(compI)
                    // println(connectedComps.length + " connected components; " + component.nodes.size + " nodes in this")

                    // TODO: use are_connected function
                    // Check if the node is adjacent to the component
                    if ((adjNodeSet intersect component.nodes).nonEmpty) {
                        // println("Connected to existing component")
                        if (nodeAddedTo == -1) {
                            // If new node is adjacent to the component and hasn't been added to another component, 
                            // add it to this component & update voxel count
                            connectedComps(compI).nodes += nodeI;
                            connectedComps(compI).voxelCount += node.voxelCount;
                            nodeAddedTo = compI;
                            // println("Node added to existing component.")
                        } else {
                            // Node has already been added to another component, but is also adjacent to this
                            // component, so the components should be merged
                            val nodeAddedToList = connectedComps(nodeAddedTo)
                            connectedComps(nodeAddedTo).nodes ++= component.nodes
                            connectedComps(nodeAddedTo).voxelCount += component.voxelCount

                            // Preserve compI after erasing component
                            connectedComps.remove(compI) // Remove old component, after merging
                            compI -= 1
                            // println("Merging components.")
                        }
                    }
                    compI += 1
                } // End connected component iteration

                if (nodeAddedTo == -1) { // If node not adjacent to any connected component, add singleton to connected components
                    val newComp = new Component(mutable.Set(nodeI), node.voxelCount)
                    connectedComps += newComp
                    // println("Node added as a singleton component.")
                }
            }
        } // End node iteration

        return connectedComps
    }
	
	private case class ImageInfo(id: Integer, fileName: String, layer: Integer, seed: (Int, Int, Int), windowing: (Int, Int))
	
	private case class XRayInfo(id: Integer, gt: String, seed: (Int, Int, Int))
	
}


