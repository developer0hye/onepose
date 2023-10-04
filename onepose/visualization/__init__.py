import cv2

def visualize_keypoints(img, 
                        keypoints, 
                        keypoint_info, 
                        skeleton_info, 
                        confidence_threshold=.5, 
                        radius=3,
                        skeleton_thickness=2,
                        keypoint_thickness=-1,
                        visualize_point_name=False):
    
    assert len(keypoints['points']) == len(keypoints['confidence']) == len(keypoint_info)
    
    points = keypoints['points']
    points_confidence = keypoints['confidence']
    
    # Create a mapping from keypoint names to points and confidence values
    name_to_point = {keypoint_info[i]['name']: pt for i, pt in enumerate(points)}
    name_to_confidence = {keypoint_info[i]['name']: pt_conf for i, pt_conf in enumerate(points_confidence)}
    
    for i, (pt, pt_confidence) in enumerate(zip(points, points_confidence)):
        if pt_confidence < confidence_threshold:
            continue
        
        start = keypoint_info[i]['name']
        start_pt = pt
        for link_info in skeleton_info.values():
            if link_info['link'][0] != start:
                continue
            end = link_info['link'][1]

            # Search for the corresponding end point in the name_to_point mapping
            end_pt = name_to_point.get(end, None)
            end_confidence = name_to_confidence.get(end, 0)
            
            # Check if the confidence of the end point is above the threshold
            if end_pt is not None and end_confidence >= confidence_threshold:
                cv2.line(img, 
                         (int(start_pt[0]), int(start_pt[1])), 
                         (int(end_pt[0]), int(end_pt[1])), 
                         link_info['color'], 
                         skeleton_thickness)
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, keypoint_info[i]['color'], keypoint_thickness)

    return img