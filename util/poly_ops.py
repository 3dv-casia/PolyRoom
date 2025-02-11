"""
Utilities for polygon manipulation.
"""
import torch
import numpy as np
import math
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def uniformencode(vertices, m):
        classes = torch.zeros(m).to("cuda")
        
        perimeter = 0
        segment_lengths = []
        newvertices = []
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            dx = vertices[i][0] - vertices[j][0]
            dy = vertices[i][1] - vertices[j][1]
            l = math.sqrt(dx**2 + dy**2)
            if l ==0:
                continue
            perimeter += l
            segment_lengths.append(l)
            newvertices.append(vertices[i])
        vertices = newvertices

        interval_length = perimeter / m

        current_length = 0
        current_segment_index = 0
        current_segment_length = segment_lengths[0]
        result_points = [torch.tensor(vertices[0], dtype = torch.float32)]
        for i in range(1, m):
        
            target_length = i * interval_length
            while current_length + current_segment_length < target_length:
                current_length += current_segment_length
                current_segment_index += 1
                current_segment_length = segment_lengths[current_segment_index % len(segment_lengths)]
            remainder_length = target_length - current_length
            alpha = remainder_length / current_segment_length
            x = (1 - alpha) * vertices[current_segment_index][0] + alpha * vertices[(current_segment_index+1) % len(vertices)][0]
            y = (1 - alpha) * vertices[current_segment_index][1] + alpha * vertices[(current_segment_index+1) % len(vertices)][1]
            result_points.append(torch.tensor([x, y], dtype = torch.float32).to("cuda"))
        point_distan = 0
        point_distans = []
        before = 0
        for i in range(1, m):
            point_distans.append(i*interval_length)
        for j in range(1, len(vertices)):
            point_distan = point_distan + segment_lengths[j - 1]
            mindiff = 10000
            for k in range(before, m - 1):
                diff = abs(point_distan - point_distans[k])
                if diff < mindiff:
                # if diff < mindiff and classes[k] != 1:
                    mindiff = diff
                    place = k
            result_points[place + 1] = torch.tensor(vertices[j])
            classes[place + 1] = int(1) 
            before = place + 1
            
        classes[0] = int(1)
        
        return result_points, classes


def unicode_instance(instance):
    room_corners = []
    corner_labels = []

    for i, poly in enumerate(instance.gt_masks.polygons):
        corners = torch.from_numpy(poly[0])
        corners = torch.clip(corners, 0, 255) / 255
        
        corners = corners.view(-1, 2)
        
        corners_pad , labels_pad = uniformencode(corners,40)
       
        corners_pad = torch.cat(corners_pad, dim = 0).to(dtype = torch.float32)
        
        room_corners.append(corners_pad)
        corner_labels.append(labels_pad)

    return torch.stack(room_corners), torch.stack(corner_labels)

def dp_unicode_instance(dpcorners):
    room_corners = []
    corner_labels = []
    if len(dpcorners) != 0:
        for i in range(len(dpcorners)):
            corners = torch.from_numpy(dpcorners[i])
            corners = torch.clip(corners, 0, 255) / 255
            
            corners_pad , labels_pad = uniformencode(corners, 40)
        
            corners_pad = torch.cat(corners_pad, dim = 0).to(dtype = torch.float32)

            room_corners.append(corners_pad)
            corner_labels.append(labels_pad)
    else:
        corners = torch.tensor([[0.25,0.25],[0.25,0.75],[0.75,0.75],[0.75,0.25]])
        
        corners_pad , labels_pad = uniformencode(corners, 40)
        corners_pad = torch.cat(corners_pad, dim = 0).to(dtype = torch.float32)

        room_corners.append(corners_pad)
        corner_labels.append(labels_pad)
    return torch.stack(room_corners), torch.stack(corner_labels)


def is_clockwise(points):
    """Check whether a sequence of points is clockwise ordered
    """
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0

def resort_corners(corners):
    """Resort a sequence of corners so that the first corner starts
       from upper-left and counterclockwise ordered in image
    """
    corners = corners.reshape(-1, 2)
    x_y_square_sum = corners[:,0]**2 + corners[:,1]**2 
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise (counterclockwise in image)
    if not is_clockwise(corners_sorted[:,:2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners_sorted.reshape(-1)


# def get_all_order_corners(corners):
#     """Get all possible permutation of a polygon
#     """
#     length = int(len(corners) / 2)
#     all_corners = torch.stack([corners.roll(i*2) for i in range(length)])
#     return all_corners

def get_all_order_corners(corners):
    """Get all possible permutation of a polygon
    """
    # length = int(len(corners) / 2)
    # all_corners = torch.stack([corners.roll(i*2) for i in range(length)])
    
    all_corners = torch.stack([corners])
    
    return all_corners


def pad_gt_polys(gt_instances, num_queries_per_poly, device):
# def pad_gt_polys(gt_instances, gt_label_instances_corners, gt_label_instances_labels, num_queries_per_poly, device):
    """Pad the ground truth polygons so that they have a uniform length
    """

    room_targets = []
    num = 0
    # padding ground truth on-fly
    for gt_inst in gt_instances:
        room_dict = {}
        room_corners = []
        corner_labels = []
        corner_lengths = []
        corners2 = []
        for i, poly in enumerate(gt_inst.gt_masks.polygons):
            corners = poly[0].reshape(-1, 2)
            newcorners = []
            delete_list = []
       
            if corners.shape[0] > 4:
            
                for k in range(corners.shape[0]):
                    ju1 = (distance(corners[k],corners[(k+1)%corners.shape[0]]) < 0 and abs(distance(corners[k-1],corners[(k)%corners.shape[0]])-distance(corners[(k+1)%corners.shape[0]],corners[(k+2)%corners.shape[0]]))<5 and abs(calculate_angles3((corners[k-1]-corners[(k)%corners.shape[0]]), (corners[(k+1)%corners.shape[0]]-corners[(k+2)%corners.shape[0]])) - math.cos(math.pi))< 0.1) 
                    ju2 = (distance(corners[k],corners[(k+1)%corners.shape[0]]) < 5 and abs(distance(corners[k-1],corners[(k)%corners.shape[0]])-distance(corners[(k+1)%corners.shape[0]],corners[(k+2)%corners.shape[0]]))<5 and abs(calculate_angles3((corners[k-1]-corners[(k)%corners.shape[0]]), (corners[(k+1)%corners.shape[0]]-corners[(k+2)%corners.shape[0]])) - math.cos(math.pi))< 0.1) and (distance(corners[k-1],corners[(k)%corners.shape[0]]) > 10 *distance(corners[k],corners[(k+1)%corners.shape[0]])) and (distance(corners[(k+1)%corners.shape[0]],corners[(k+2)%corners.shape[0]]) > 10 *distance(corners[k],corners[(k+1)%corners.shape[0]]))
                    if ju1 or ju2:
                        
                        delete_list.append(k)
                        delete_list.append(k+1)
                        continue

                for k in range(corners.shape[0]):
                    if k in delete_list:
                        continue
                    newcorners.append(corners[k])
                corners = np.array(newcorners)
                newcorners = []
                delete_list = []
                for k in range(corners.shape[0]):
                    ju1 = (distance(corners[k],corners[(k+1)%corners.shape[0]]) < 0 and abs(distance(corners[k-1],corners[(k)%corners.shape[0]])-distance(corners[(k+1)%corners.shape[0]],corners[(k+2)%corners.shape[0]]))<5 and abs(calculate_angles3((corners[k-1]-corners[(k)%corners.shape[0]]), (corners[(k+1)%corners.shape[0]]-corners[(k+2)%corners.shape[0]])) - math.cos(math.pi))< 0.1) 
                    ju2 = (distance(corners[k],corners[(k+1)%corners.shape[0]]) < 5 and abs(distance(corners[k-1],corners[(k)%corners.shape[0]])-distance(corners[(k+1)%corners.shape[0]],corners[(k+2)%corners.shape[0]]))<5 and abs(calculate_angles3((corners[k-1]-corners[(k)%corners.shape[0]]), (corners[(k+1)%corners.shape[0]]-corners[(k+2)%corners.shape[0]])) - math.cos(math.pi))< 0.1) and (distance(corners[k-1],corners[(k)%corners.shape[0]]) > 10 *distance(corners[k],corners[(k+1)%corners.shape[0]])) and (distance(corners[(k+1)%corners.shape[0]],corners[(k+2)%corners.shape[0]]) > 10 *distance(corners[k],corners[(k+1)%corners.shape[0]]))
                    if ju1 or ju2:
                        delete_list.append(k)
                        delete_list.append(k+1)
                        continue

                for k in range(corners.shape[0]):
                    if k in delete_list:
                        continue
                    newcorners.append(corners[k])
                corners = np.array(newcorners)
                newcorners =  []
                for k in range(corners.shape[0]):
                    if abs(calculate_angles2(corners[(k-1)%corners.shape[0]],corners[k],corners[(k+1)%corners.shape[0]]))> math.cos(1*math.pi/9):
                        continue
                    newcorners.append(corners[k])
                corners = torch.tensor(newcorners).to(device)
                
            else:
                corners = torch.tensor(corners).to(device)
            corners = torch.clip(corners, 0, 255) / 255
          
            corners_pad , labels_pad = uniformencode(corners, num_queries_per_poly)
            corners_pad = torch.cat(corners_pad, dim = 0).to(dtype = torch.float32)
            corner_lengths.append(len(corners_pad))
        
            room_corners.append(corners_pad)
            corner_labels.append(labels_pad)
   
            
        room_dict = {
            'lengths': torch.tensor(corner_lengths, device=device),
            'room_labels': gt_inst.gt_classes
        }
        room_dict['coords'] = torch.stack(room_corners)
        room_dict['labels'] = torch.stack(corner_labels)
        room_targets.append(room_dict)
        
        num = num + 1
    # room_targets = []
    # # padding ground truth on-fly
    # for gt_inst in gt_instances:
    #     room_dict = {}
    #     room_corners = []
    #     corner_labels = []
    #     corner_lengths = []

    #     for i, poly in enumerate(gt_inst.gt_masks.polygons):
    #         corners = torch.from_numpy(poly[0]).to(device)
    #         corners = torch.clip(corners, 0, 255) / 255
    #         corner_lengths.append(len(corners))

    #         corners_pad = torch.zeros(num_queries_per_poly*2, device=device)
    #         corners_pad[:len(corners)] = corners

    #         labels = torch.ones(int(len(corners)/2), dtype=torch.int64).to(device)
    #         labels_pad = torch.zeros(num_queries_per_poly, device=device)
    #         labels_pad[:len(labels)] = labels
    #         room_corners.append(corners_pad)
    #         corner_labels.append(labels_pad)

    #     room_dict = {
    #         'coords': torch.stack(room_corners),
    #         'labels': torch.stack(corner_labels),
    #         'lengths': torch.tensor(corner_lengths, device=device),
    #         'room_labels': gt_inst.gt_classes
    #     }
    #     room_targets.append(room_dict)


    return room_targets

def calculate_angles2(p1,p2,p3):
    
    vect1 = torch.tensor(p2-p1).float()
    vect2 = torch.tensor(p2-p3).float()
  
    cos_sim = ((vect1 * vect2).sum(-1)+1e-9)/(torch.norm(vect1, p=2)*torch.norm(vect2, p=2)+1e-9)
    # cos_sim = torch.clamp(cos_sim, min=-1, max=1)
    # angles = torch.acos(cos_sim)
    # return angles
    cos_sim = cos_sim
    return cos_sim

def calculate_angles3(vec1,vec2):
    
    vect1 = torch.tensor(vec1).float()
    vect2 = torch.tensor(vec2).float()
  
    cos_sim = ((vect1 * vect2).sum(-1)+1e-9)/(torch.norm(vect1, p=2)*torch.norm(vect2, p=2)+1e-9)
    # cos_sim = torch.clamp(cos_sim, min=-1, max=1)
    # angles = torch.acos(cos_sim)
    # return angles
    cos_sim = cos_sim
    return cos_sim


