# Pseudocode for Generating Heatmaps

import numpy as np

def generate_heatmaps(classes, view_imgs, proto_acts):
    all_proto_acts_max = []
    # for all classes get the top three prototype weights
    for category_id in range(classes):
        all_prototypes = get_all_prototypes_for_class(category_id)
        top_three_prototypes_for_class = all_prototypes.max(axis=(1, 2))
        all_proto_acts_max.append(top_three_prototypes_for_class)

    # Combine all proto_acts_max values into a single array and then into a single value
    all_proto_acts_max = np.concatenate(all_proto_acts_max)
    total_weight = np.sum(all_proto_acts_max)

    # Calculate weighted heatmaps per image 
    for image in view_imgs:
        for class_id in range(classes):
            combined_heatmap = initialize_heatmap()

            # Get top 3 prototypes for current class and image
            top3_proto_indices = get_top3_prototypes(image, class_id)

            # Overlay heatmaps of top 3 prototypes
            for proto_idx in top3_proto_indices:
                heatmap = generate_heatmap_for_prototype(image, proto_idx)
                # get the weight for the current prototype
                weight = all_proto_acts_max[proto_idx] / total_weight
                # scale the heatmap bei the weigth
                scaled_heatmap = scale_heatmap(heatmap, weight)
                combined_heatmap += scaled_heatmap

            # overlay heatmap with image and save 


