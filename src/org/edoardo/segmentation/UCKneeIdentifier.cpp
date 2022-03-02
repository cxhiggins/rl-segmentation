/* 
 * File:   UCKneeIdentifier.cpp
 * Author: Cara Higgins
 * 
 * Created on 20 October 2021, 16:12
 */

#include "UCKneeIdentifier.h"

#include <climits>

#include <boost/bind.hpp>

#include <common/partitionforests/images/GeneralFeature.h>
#include <common/dicom/volumes/DICOMVolume.h>
#include <common/util/ITKImageUtil.h>

namespace mp {

    //#################### CONSTRUCTORS ####################

    UCKneeIdentifier::UCKneeIdentifier(const DICOMVolume_CPtr& dicomVolume, const VolumeIPF_Ptr& volumeIPF)
    : FeatureIdentifier(dicomVolume, volumeIPF) {
    }

    //#################### PUBLIC METHODS ####################

    int UCKneeIdentifier::length() const {
        return 5;
    }

    //#################### PRIVATE METHODS ####################

    std::tuple<std::list<PFNodeID>, int> ucknee;

    // /**
    // @brief	Select region of interest (ROI) by setting window minima and maxima

    // @param[in]	volumeSize	    [x width, y height, z count]
    // @param[in]  distFromEdge    User-selected parameter indicating the minimum distance a node must be from the edge; 
    //                             prevents selection of border nodes, which often have very bright values
    // @param[in]  windMinX        Pointer to an integer variable representing the minimum x-value for the ROI
    // @param[in]  windMinY
    // @param[in]  windMaxX
    // @param[in]  windMaxY
    // */
    // void selectROI(itk::Size<3> volumeSize, int distFromEdge, int* windMinX, int* windMinY, int* windMaxX, int* windMaxY) {
    //     int width = volumeSize[0];
    //     int height = volumeSize[1];
    //     float htow = height * 1.0 / width;

    //     *windMinY = distFromEdge;
    //     *windMinX = distFromEdge;
    //     *windMaxX = width-distFromEdge;
    //     *windMaxY = height-distFromEdge;

    //     // TODO: check that distFromEdge <= width / 4, height / 4
    //     // Determine region of focus based on image size
    //     if (0.75 < htow && htow < 1.5) { // Roughly square
    //         *windMinX = width / 4;
    //         *windMaxX = width * 3 / 4;

    //         *windMinY = height / 4;
    //         *windMaxY = height * 3 / 4;
    //     } else if (htow > 1.5) {
    //         *windMinY = height / 4;
    //         *windMaxY = height * 3 / 4;
    //     } else {
    //         *windMinX = width / 4;
    //         *windMaxX = width * 3 / 4;
    //     }

    //     std::cout<<"Height to weight ratio: "<<htow<<std::endl;
    //     std::cout<<"image (height x width) = "<<height<<" x "<<width<<std::endl;
    //     std::cout<<"window = ("<<*windMinX<<", "<<*windMinY<<") -> ("<<*windMaxX<<", "<<*windMaxY<<")"<<std::endl;
    // }

    void UCKneeIdentifier::execute_impl() {
        set_status("Identifying unicompartmental knee...");
        std::cout<<"OnMenuSegmentationSegmentUCKnee reached."<<std::endl;

        int windMinX, windMinY, windMaxX, windMaxY;
        // TODO: only call if ROI is not already initialized
        selectROI(dicom_volume()->size(), 10, &windMinX, &windMinY, &windMaxX, &windMaxY);        

        std::vector<int> node_indices = volume_ipf()->node_indices(layerIndex);
        std::cout<<"Nodes in layer "<<layerIndex<<" = "<<node_indices.size()<<std::endl;
        
        std::tuple<std::list<PFNodeID>, int> nodesWithMaxMeanGray = max_mean_grey_with_filter(
            node_indices, windMinX, windMinY, windMaxX, windMaxY
        );

        std::list<PFNodeID> nodes = std::get<0>(nodesWithMaxMeanGray);
        int maxMeanGrey = std::get<1>(nodesWithMaxMeanGray);
        
        int threshold = maxMeanGrey - 20;

        std::list<component> connectedComps = connected_components(nodes, threshold);
        connectedComps.sort(compare_components);    // Sort connected components by voxel count

        // Initialize region to store unicompartmental knee nodes
        PartitionForestSelection_Ptr filledRegion(new PartitionForestSelectionT(volume_ipf()));

        // Select the two largest components and set the ucknee variable accordingly
        for (int i = 0; i < 2 && !connectedComps.empty(); i++) {
            component comp = connectedComps.front();
            connectedComps.pop_front();

            std::set<PFNodeID> compNodes(std::get<0>(comp).begin(), std::get<0>(comp).end());

            std::cout<<"calling merge_nonsibling_nodes"<<std::endl;
            compNodes = volume_ipf()->merge_nonsibling_nodes(compNodes, volume_ipf()->CheckPreconditions::CHECK_PRECONDITIONS);
            std::cout<<"merge_nonsibling_nodes executed"<<std::endl;
            PFNodeID compNode = *compNodes.begin();
            // PFNodeID compNode = volume_ipf()->merge_sibling_nodes(compNodes, volume_ipf()->CheckPreconditions::CHECK_PRECONDITIONS);

            ucknee.push_back(compNode);
            filledRegion->select_node(compNode);
        }

        VolumeIPFMultiFeatureSelection_Ptr multiFeatureSelection = get_multi_feature_selection();
        multiFeatureSelection->identify_selection(filledRegion, GeneralFeature::UC_KNEE);
        std::cout<<"Selected top 2 connected components"<<std::endl;
        get_bounding_boxes();
    }

    /**
    @brief	Compares two components based on their associated voxel count.

    @param[in]	first	The first component
    @param[in]  second  The second component

    @return	true if first has a voxel count greater than that of the second component.
    */
    bool UCKneeIdentifier::compare_components(const std::tuple<std::list<PFNodeID>, int>& first, const std::tuple<std::list<PFNodeID>, int>& second)
    {
        return std::get<1>(first) > std::get<1>(second);
    }
    
    /**
    @brief	Prints the bounding boxes for the two components of the UC knee prosthesis
    */
    void UCKneeIdentifier::get_bounding_boxes()
    {
        for (std::list<PFNodeID>::iterator it = ucknee.begin(); it != ucknee.end(); it++) {
            propertyMap propMap = volume_ipf()->branch_properties(*it).branch_property_map();
            
            int MinX = std::stoi(propMap.at("X Min"));
            int MinY = std::stoi(propMap.at("Y Min"));
            int MaxX = std::stoi(propMap.at("X Max"));
            int MaxY = std::stoi(propMap.at("Y Max"));

            std::cout<<"Bounding box: ("<<MinX<<", "<<MinY<<") -> ("<<MaxX<<", "<<MaxY<<")"<<std::endl;
        }
    }

    // std::tuple<std::list<PFNodeID>, int> UCKneeIdentifier::max_mean_grey_with_filter(
    //     std::vector<int> node_indices,
    //     int windMinX, int windMinY, int windMaxX, int windMaxY
    // ) {   
    //     int meanGrey; int VoxelCount; int MinX; int MinY; int MaxX; int MaxY; 
    //     int maxMeanGrey = 0;
    //     std::list<PFNodeID> nodes;
        
    //     for(std::vector<int>::iterator indexIt = node_indices.begin(); indexIt != node_indices.end(); indexIt++)
    //     {
    //         PFNodeID node = PFNodeID(layerIndex, *indexIt);
    //         propertyMap propMap = volume_ipf()->branch_properties(node).branch_property_map();

    //         meanGrey = std::stof(propMap.at("Mean Grey Value"));
    //         VoxelCount = std::stoi(propMap.at("Voxel Count"));
    //         MinX = std::stoi(propMap.at("X Min"));
    //         MinY = std::stoi(propMap.at("Y Min"));
    //         MaxX = std::stoi(propMap.at("X Max"));
    //         MaxY = std::stoi(propMap.at("Y Max"));

    //         if (
    //             VoxelCount <= 15000 && VoxelCount >= 10 && // Node must contain between 100 and 15,000 voxels
    //             windMinX < MinX && MinX < windMaxX && windMinY < MinY && MinY < windMaxY // Region lies within region of interest (not touching edges)
    //         ) { 
    //             nodes.push_back(node);  // Add node to list of filtered nodes

    //             if (meanGrey > maxMeanGrey)
    //                 maxMeanGrey = meanGrey;
    //         }
    //     }

    //     std::cout<<"Max Mean Grey Value = "<<maxMeanGrey<<std::endl;

    //     return std::tuple<std::list<PFNodeID>, int>(nodes, maxMeanGrey);
    // }

    std::list<UCKneeIdentifier::component> UCKneeIdentifier::connected_components(std::list<PFNodeID> nodes, int threshold)
    {
        std::list<component> connectedComps;

        for (std::list<PFNodeID>::iterator it = nodes.begin(); it != nodes.end(); it++) {
            propertyMap propMap = volume_ipf()->branch_properties(*it).branch_property_map();
            int meanGrey = std::stof(propMap.at("Mean Grey Value"));

            if (meanGrey > threshold) { // Check that mean grey value is above the specified threshold 
                PFNodeID node = *it;
                component* nodeAddedTo = nullptr; // Pointer to the component the node has been added to

                // Obtain set of adjacent nodes
                std::vector<int> adjNodes = volume_ipf()->adjacent_nodes(node);
                std::set<int> adjNodeSet(adjNodes.begin(), adjNodes.end());

                // Iterate through connected components
                for (std::list<component>::iterator compIt = connectedComps.begin(); compIt != connectedComps.end(); compIt++) {
                    std::list<PFNodeID> compNodes = std::get<0>(*compIt);

                    std::cout<<connectedComps.size()<<" connected components; "<<compNodes.size()<<" nodes in this"<<std::endl;

                    // TODO: use are_connected function
                    // Convert compNodes to a set of node indices
                    std::set<int> nodes;
                    for (std::list<PFNodeID>::iterator nodeIt = compNodes.begin(); nodeIt != compNodes.end(); nodeIt++) {
                        nodes.insert(nodeIt->index());
                    }

                    // Check if the node is adjacent to the component
                    if (volume_ipf()->are_connected(nodes, layerIndex)) {
                        std::cout<<"Connected"<<std::endl;
                        if (!nodeAddedTo) {
                            // If new node is adjacent to the component and hasn't been added to another component, 
                            // add it to this component & update voxel count
                            compNodes.push_back(node);
                            std::get<0>(*compIt) = compNodes;
                            std::get<1>(*compIt) += std::stoi(propMap.at("Voxel Count"));
                            nodeAddedTo = &(*compIt);
                            std::cout<<"Node added to existing component."<<std::endl;
                        } else {
                            // Node has already been added to another component, but is also adjacent to this
                            // component, so the components should be merged
                            std::list<PFNodeID> nodeAddedToList = std::get<0>(*nodeAddedTo);
                            nodeAddedToList.splice(nodeAddedToList.begin(), compNodes);
                            std::get<0>(*nodeAddedTo) = nodeAddedToList;
                            std::get<1>(*nodeAddedTo) += std::get<1>(*compIt);

                            // Preserve compIt after erasing component
                            connectedComps.erase(compIt++); // Remove old component, after merging
                            compIt--;
                            std::cout<<"Merging components."<<std::endl;
                        }
                    }
                } // End connected component iteration

                if (!nodeAddedTo) { // If node not adjacent to any connected component, add singleton to connected components
                    component newComp = component(std::list<PFNodeID>(1, node), std::stoi(propMap.at("Voxel Count")));
                    connectedComps.push_back(newComp);
                    std::cout<<"Node added as a singleton component."<<std::endl;
                }
            }
        } // End node iteration

        return connectedComps;
    }

    UCKneeIdentifier::BranchProperties UCKneeIdentifier::calculate_component_properties(int layer, const std::set<int>& indices) const
    {
        std::vector<BranchProperties> nodeProperties;
        nodeProperties.reserve(indices.size());
        for(std::set<int>::const_iterator it=indices.begin(), iend=indices.end(); it!=iend; ++it)
        {
            PFNodeID node(layer, *it);
            nodeProperties.push_back(volume_ipf()->branch_properties(node));
        }
        return BranchProperties::combine_branch_properties(nodeProperties);
    }

    // std::set<PFNodeID> decompose_ucknee()
    // {

    //     // return split_node(const PFNodeID& node, const std::vector<std::set<int> >& groups, CheckPreconditions checkPreconditions = CHECK_PRECONDITIONS)
    // }
}