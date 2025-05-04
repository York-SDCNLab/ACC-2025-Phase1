import heapq
from scipy.interpolate import interp1d
from typing import Dict, Union, Tuple, List, Set

import cv2
import torch
import numpy as np

from hal.utilities.path_planning import RoadMap, RoadMapNode

from .constants import X_OFFSET, Y_OFFSET, ACC_SCALE
from .constants import NODE_POSES_RIGHT_COMMON
from .constants import NODE_POSES_RIGHT_LARGE_MAP
from .constants import EDGE_CONFIGS_RIGHT_COMMON
from .constants import EDGE_CONFIGS_RIGHT_LARGE_MAP


class ACCRoadMap(RoadMap):
    """
    This class is responsible for generating waypoints of the roadmap used in
    the ACC2024 student self-driving competition
    """
    def __init__(self, scale: float = ACC_SCALE) -> None:
        """
        Initializes the ACCRoadMap object
        """
        # parent class initialization
        super().__init__()
        # read nodes and edges
        node_positions: list = NODE_POSES_RIGHT_COMMON + NODE_POSES_RIGHT_LARGE_MAP
        edges: list = EDGE_CONFIGS_RIGHT_COMMON + EDGE_CONFIGS_RIGHT_LARGE_MAP
        # add scaled nodes to acc map
        for index, position in enumerate(node_positions):  # [1134, 1454, -HALF_PI]
            position[0] = scale * (position[0] - X_OFFSET)
            position[1] = scale * (Y_OFFSET - position[1])
            self.add_node(position, index)
        # add scaled edge to acc map
        for edge in edges:
            edge[2] = edge[2] * scale
            self.add_edge(*edge)

    def generate_random_cycle(self, start: int, min_length:int = 3) -> list:
        """
        Generates a random cycle from a given starting node

        Parameters:
        - start: int: The starting node
        - min_length: int: The minimum length of the cycle

        Returns:
        - list: The list of nodes in the cycle
        """
        # depth first search for finding all cycles that start and end at the starting point
        def dfs(start):
            fringe: list = [(start, [])]

            while fringe:
                node, path = fringe.pop()
                if path and node == start:
                    yield path
                    continue
                for next_edges in node.outEdges:
                    next_node = next_edges.toNode
                    if next_node in path:
                        continue
                    fringe.append((next_node, path + [next_node]))

        start_node: RoadMapNode = self.nodes[start]
        cycles: list = [[start_node] + path for path in dfs(start_node) if len(path) > min_length]
        num_cycles: int = len(cycles)
        return cycles[np.random.randint(num_cycles)]

    def generate_path(self, node_sequence: Union[np.ndarray, list]) -> np.array:
        """
        Wraps the generated path as a numpy array object

        Parameters:
        - node_sequence: Union[np.ndarray, list]: The sequence of nodes

        Returns:
        - np.array: The path as a numpy array
        """
        if type(node_sequence) == np.ndarray:
            node_sequence = node_sequence.tolist()

        return np.array(super().generate_path(node_sequence)).transpose(1, 0) #[N, (x, y)]

    def generate_path_and_segments(self, node_sequence: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wraps the generated path and segments as a numpy array object

        Parameters:
        - node_sequence: Union[np.ndarray, list]: The sequence of nodes

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The path and segments as numpy arrays
        """
        if type(node_sequence) == np.ndarray:
            node_sequence = node_sequence.tolist()

        # convert from map node to index
        sequence_ids = node_sequence
        if not type(sequence_ids[0]) == int:
            sequence_ids = [node.index for node in sequence_ids]

        # generate path and find which waypoints belong to which road segment
        path = []
        segments = {}
        waypoint_index = 0
        for i in range(1, len(sequence_ids)):
            sub_sequence = sequence_ids[i-1:i+1]
            sub_sequence_path = np.array(super().generate_path(sub_sequence)).transpose(1, 0) #[N, (x, y)]
            sub_sequence_length = sub_sequence_path.shape[0]

            path.append(sub_sequence_path)
            segments[(waypoint_index, waypoint_index + sub_sequence_length - 1)] = np.array(sub_sequence)
            waypoint_index += sub_sequence_length

        path = np.vstack(path)
        return path, segments

    def prepare_map_info(self, node_sequence: list) -> Tuple[dict, np.ndarray]:
        """
        Provide the position informations related to the node sequence

        Parameters:
        - node_sequence: Union[np.ndarray, list]: The sequence of nodes

        Returns:
        - Tuple[dict, np.ndarray]: The list of nodes' and waypoints' position
        """
        node_dict: Dict[str, np.ndarray] = {}
        for node_id in node_sequence:
            pose: np.ndarray = self.nodes[node_id].pose
            node_dict[node_id] = pose # x, y, angle

        waypoint_sequence = self.generate_path(node_sequence)
        return node_dict, waypoint_sequence
    
    def find_shortest_path(self, startNode, goalNode):
        """Find the shortest path between two nodes using the A* algorithm.

        Args:
            startNode (int or RoadMapNode): Starting node (index or instance).
            goalNode (int or RoadMapNode): Goal node (index or instance).
            radius (float): Minimum turning radius.

        Returns:
            path: generated path as a 2xn numpy array
        """
        if startNode == goalNode:
            return None

        if type(startNode) == int:
            startNode = self.nodes[startNode]
        if type(goalNode) == int:
            goalNode = self.nodes[goalNode]

        # Initialize the open set and closed set
        openSet = []
        closedSet = set()

        # Add the start node to the open set with a cost of 0 and an
        # f-score equal to the heuristic estimate
        heapq.heappush(
            openSet,
            (0 + self._heuristic(startNode, goalNode), startNode)
        )

        # Initialize the g-scores for each node to infinity
        gScore = {node: float('inf') for node in self.nodes}
        gScore[startNode] = 0

        # Initialize the 'came from' (node, edge) pair
        # for each node to None
        cameFrom = {node: None for node in self.nodes}

        while openSet:
            # Pop the node with the lowest f-score from the open set
            currentNode = heapq.heappop(openSet)[1]

            if currentNode == goalNode:
                # Goal node found, construct the optimal path, then return
                path = goalNode.pose[:2,:]
                node = goalNode
                index_path = np.array([node.index])
                while True:
                    (node, edge) = cameFrom[node]
                    index_path = np.concatenate([index_path, [node.index]])
                    path = np.hstack((
                        node.pose[:2,:],
                        edge.waypoints,
                        path
                    ))
                    if cameFrom[node] is None:
                        break

                return path, index_path

            closedSet.add(currentNode)

            for edge in currentNode.outEdges:
                neighborNode = edge.toNode
                if neighborNode in closedSet:
                    # Neighbor node already explored
                    continue

                if edge.length is None:
                    tentative_g = float('inf')
                else:
                    tentative_g = gScore[currentNode] + edge.length

                if tentative_g < gScore[neighborNode]:
                    # New path to neighbor node found,
                    # update parent pointer and g-score
                    cameFrom[neighborNode] = (currentNode, edge)
                    gScore[neighborNode] = tentative_g

                    # Add the neighbor node to the open set with a cost
                    # equal to the g-score plus the heuristic estimate
                    hScore = self._heuristic(neighborNode, goalNode)
                    heapq.heappush(
                        openSet,
                        (gScore[neighborNode] + hScore, neighborNode)
                    )

        # Open set is empty and goal node not found, no path exists
        return None


class CustomRoadMap(RoadMap): 
    def __init__(
        self,
        n_mp_pl: int = 1024,
        n_mp_pl_node: int = 30,
        removed_node_list: list = [],
    ): 
        # parent class initialization 
        super().__init__()
        # read nodes and edges 
        node_positions = NODE_POSES_RIGHT_COMMON + NODE_POSES_RIGHT_LARGE_MAP 
        edges = EDGE_CONFIGS_RIGHT_COMMON + EDGE_CONFIGS_RIGHT_LARGE_MAP  
        # add scaled nodes to acc map 
        for index, position in enumerate(node_positions): 
            position[0] = ACC_SCALE * (position[0] - X_OFFSET) 
            position[1] = ACC_SCALE * (Y_OFFSET - position[1]) 
            self.add_node(position, index) 
        # add scaled edge to acc map 
        for edge in edges: 
            edge[2] = edge[2] * ACC_SCALE 
            self.add_edge(*edge)

        #create polylines for rendering
        self.agent_length = 0.4
        self.agent_width = 0.2
        self.road_width = 0.27
        self.removed_node_list = removed_node_list

        self.map_polylines = []

        # each edge waypoint is spaced 1 cm apart, so if n_mp_pl_node = 30 then
        # each polyline represents a 30cm road segment (a bit smaller than the)
        # size of the qcar
        self.n_mp_pl = n_mp_pl #number of polylines in the map
        self.n_mp_pl_node = n_mp_pl_node #number of nodes in a polyline
        self.map_valid = np.zeros([self.n_mp_pl, self.n_mp_pl_node], dtype=bool)
        self.map_id = np.zeros([self.n_mp_pl], dtype=np.int64) - 1
        self.map_pos = np.zeros([self.n_mp_pl, self.n_mp_pl_node, 2], dtype=np.float32)
        self.map_dir = np.zeros([self.n_mp_pl, self.n_mp_pl_node, 2], dtype=np.float32)

        #smart map format, treat all as road edge for qlabs TODO: add crosswalk
        self.map_infos = {
            #"lanes": [],
            "road_edge": [],
            #"road_lines": [],
            #"crosswalk": []
        }
        
        self.map_counter = 0 #n_mp
        self.point_count = 0
        for i, edge in enumerate(self.edges):
            if edge.waypoints is None:
                continue

            waypoints = np.transpose(edge.waypoints) #pl_pos
            dw = np.diff(waypoints, axis=0) #pl_dir
            theta = np.arctan2(dw[:, 1], dw[:, 0])
            theta = np.concatenate([[theta[0]], theta])
            right_lane = waypoints + np.transpose(np.array([np.cos(theta + (np.pi/2)), np.sin(theta + (np.pi/2))]))*(self.road_width / 2)
            left_lane = waypoints + np.transpose(np.array([np.cos(theta - (np.pi/2)), np.sin(theta - (np.pi/2))]))*(self.road_width / 2)

            #segment polylines (Trafficbots)
            polyline_len = dw.shape[0]
            polyline_cuts = np.linspace(0, polyline_len, polyline_len // self.n_mp_pl_node + 1, dtype=int, endpoint=False)
            num_cuts = len(polyline_cuts)
            for idx_cut in range(num_cuts):
                idx_start = polyline_cuts[idx_cut]
                if idx_cut + 1 == num_cuts:
                    #last cut
                    idx_end = polyline_len
                else:
                    idx_end = polyline_cuts[idx_cut + 1]

                #note: there is a left and right road line for each edge, so we define 2 polylines for each edge
                self.map_valid[self.map_counter:self.map_counter+2, : idx_end - idx_start] = True
                self.map_pos[self.map_counter, : idx_end - idx_start] = right_lane[idx_start:idx_end]
                self.map_pos[self.map_counter + 1, : idx_end - idx_start] = left_lane[idx_start:idx_end]
                self.map_dir[self.map_counter:self.map_counter+2, : idx_end - idx_start] = dw[idx_start:idx_end]
                #self.map_id[self.map_counter:self.map_counter+2] = edge.fromNode.index
                #self.map_counter += 2

                #get polyline info (SMART)
                cur_info_right = {
                    "id": self.map_counter,
                    "polyline_index": (self.point_count, self.point_count + right_lane[idx_start:idx_end].shape[0])
                }
                self.point_count += right_lane[idx_start:idx_end].shape[0]

                cur_info_left = {
                    "id": self.map_counter + 1,
                    "polyline_index": (self.point_count, self.point_count + left_lane[idx_start:idx_end].shape[0])
                }
                self.point_count += left_lane[idx_start:idx_end].shape[0]

                self.map_infos["road_edge"].append(cur_info_right)
                self.map_infos["road_edge"].append(cur_info_left)
                self.map_polylines.append(right_lane)
                self.map_polylines.append(left_lane)
                self.map_counter += 2

        #concatenate polylines
        self.map_polylines = np.concatenate(self.map_polylines, axis=0).astype(np.float32)

        #parse smart map features
        polygon_ids = [x["id"] for x in self.map_infos["road_edge"]]
        num_polygons = len(polygon_ids) #should be the same as map_counter
        point_position = [None] * num_polygons

        for _seg in self.map_infos["road_edge"]:
            _idx = polygon_ids.index(_seg["id"])
            roadline = self.map_polylines[_seg["polyline_index"][0] : _seg["polyline_index"][1]]
            point_position[_idx] = roadline[:-1, :2]

        num_points = np.array([point.shape[0] for point in point_position], dtype=int)
        point_to_polygon_edge_index = np.stack([
            np.arange(num_points.sum(), dtype=int),
            np.arange(num_polygons, dtype=int).repeat(num_points),
        ], axis=0)

        point_position = np.concatenate(point_position, axis=0) #equivalent to map_data["map_point"]["position"]
        split_polyline_pos = []
        split_polyline_theta = []

        #split polylines
        for i in sorted(np.unique(point_to_polygon_edge_index[1])):
            index = point_to_polygon_edge_index[0, point_to_polygon_edge_index[1] == i]
            if len(index) <= 2:
                continue

            cur_pos = point_position[index, :2]

            #interpolate polyline and convert to pytorch tensor
            split_polyline = self._interpolate_polyline(cur_pos)
            if split_polyline is None:
                continue

            split_polyline_pos.append(split_polyline[..., :2])
            split_polyline_theta.append(split_polyline[..., 2])

        self.map_data_smart = {}
        self.map_data_smart["map_save"] = {
            "traj_pos": torch.cat(split_polyline_pos, dim=0),  # [num_nodes, 3, 2]
            "traj_theta": torch.cat(split_polyline_theta, dim=0)[:, 0],  # [num_nodes]
        }
        self.map_data_smart["pt_token"] = {
            "num_nodes": self.map_data_smart["map_save"]["traj_pos"].shape[0],
        }

        #print("NUM POLYLINES: {}".format(self.map_data_smart["map_save"]["traj_pos"].shape[0]))

        #get the closest map polyline to each high level node
        #SMART map
        self.node_pl_map = np.zeros((self.map_data_smart["map_save"]["traj_pos"].shape[0], ), dtype=np.int64)
        flat_map_pos = self.map_data_smart["map_save"]["traj_pos"][:, 0] #num_nodes, (x, y)
        self.node_positions = np.array(node_positions)
        for i in range(self.node_positions.shape[0]):
            map_node = self.node_positions[i]
            dist = np.linalg.norm(flat_map_pos - map_node[:2], axis=-1)
            self.node_pl_map[i] = dist.argmin() // n_mp_pl_node

        #Trafficbots map
        '''flat_map_pos = self.map_pos.reshape(-1, 2) #n_mp*n_mp_pl_node, 2
        self.node_positions = np.array(node_positions)
        self.node_pl_map = np.zeros((self.node_positions.shape[0], ), dtype=np.int64)
        for i in range(self.node_positions.shape[0]):
            map_node = self.node_positions[i]
            dist = np.linalg.norm(flat_map_pos - map_node[:2], axis=-1)
            self.node_pl_map[i] = dist.argmin() // n_mp_pl_node'''

        #get the full map boundary for rendering
        pos = self.map_pos[self.map_valid]
        xmin = pos[:, 0].min()
        ymin = pos[:, 1].min()
        xmax = pos[:, 0].max()
        ymax = pos[:, 1].max()
        self.map_boundary = np.array([xmin, xmax, ymin, ymax])

    def _interpolate_polyline(
        self,
        polylines: np.ndarray,
        distance: float = 0.1,
        split_distance: int = 1
    ):
        # Calculate the cumulative distance along the path, up-sample the polyline to 5 cm
        dist_along_path_list = []
        polylines_list = []
        euclidean_dists = np.linalg.norm(polylines[1:, :2] - polylines[:-1, :2], axis=-1)
        euclidean_dists = np.concatenate([[0], euclidean_dists])
        breakpoints = np.where(euclidean_dists > 3)[0]
        breakpoints = np.concatenate([[0], breakpoints, [polylines.shape[0]]])
        for i in range(1, breakpoints.shape[0]):
            start = breakpoints[i - 1]
            end = breakpoints[i]
            dist_along_path_list.append(
                np.cumsum(euclidean_dists[start:end]) - euclidean_dists[start]
            )
            polylines_list.append(polylines[start:end])

        multi_polylines_list = []
        for idx in range(len(dist_along_path_list)):
            if len(dist_along_path_list[idx]) < 2:
                continue
            dist_along_path = dist_along_path_list[idx]
            polylines_cur = polylines_list[idx]
            # Create interpolation functions for x and y coordinates
            fxy = interp1d(dist_along_path, polylines_cur, axis=0)

            # Create an array of distances at which to interpolate
            new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
            new_dist_along_path = np.concatenate(
                [new_dist_along_path, dist_along_path[[-1]]]
            )

            # Combine the new x and y coordinates into a single array
            new_polylines = fxy(new_dist_along_path)
            polyline_size = int(split_distance / distance)
            if new_polylines.shape[0] >= (polyline_size + 1):
                padding_size = (
                    new_polylines.shape[0] - (polyline_size + 1)
                ) % polyline_size
                final_index = (
                    new_polylines.shape[0] - (polyline_size + 1)
                ) // polyline_size + 1
            else:
                padding_size = new_polylines.shape[0]
                final_index = 0
            multi_polylines = None
            new_polylines = torch.from_numpy(new_polylines)
            new_heading = torch.atan2(
                new_polylines[1:, 1] - new_polylines[:-1, 1],
                new_polylines[1:, 0] - new_polylines[:-1, 0],
            )
            new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
            new_polylines = torch.cat([new_polylines, new_heading], -1)
            if new_polylines.shape[0] >= (polyline_size + 1):
                multi_polylines = new_polylines.unfold(
                    dimension=0, size=polyline_size + 1, step=polyline_size
                )
                multi_polylines = multi_polylines.transpose(1, 2)
                multi_polylines = multi_polylines[:, ::5, :]
            if padding_size >= 3:
                last_polyline = new_polylines[final_index * polyline_size :]
                last_polyline = last_polyline[
                    torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()
                ]
                if multi_polylines is not None:
                    multi_polylines = torch.cat(
                        [multi_polylines, last_polyline.unsqueeze(0)], dim=0
                    )
                else:
                    multi_polylines = last_polyline.unsqueeze(0)
            if multi_polylines is None:
                continue
            multi_polylines_list.append(multi_polylines)
        if len(multi_polylines_list) > 0:
            multi_polylines_list = torch.cat(multi_polylines_list, dim=0).to(torch.float32)
        else:
            multi_polylines_list = None
        return multi_polylines_list

    def _to_pixel(self, pos, transform=None, top_left_px = None, px_per_meter=None):
        #default px_per_meter
        if px_per_meter is None:
            px_per_meter = 192 * 0.48

        #global coordinate system
        if transform is None:
            assert top_left_px is not None, "Require provided top left pixel for gcs rendering"

            pos = pos * px_per_meter
            pos[..., 0] = pos[..., 0] - top_left_px[0]
            pos[..., 1] = -pos[..., 1] - top_left_px[1]
            return np.round(pos).astype(np.int32)

        #agent centric coordinate system
        x, y, yaw = transform
        yaw -=  np.pi / 2

        if top_left_px is None:
            top_left_px = np.array([x - 1.0, -y - (2.0 - (self.agent_length / 2))], dtype=np.float32) * px_per_meter

        rot = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])

        pos = pos - np.array([x, y])
        pos = np.matmul(pos, rot)
        pos = pos + np.array([x, y])

        pos = pos * px_per_meter
        pos[..., 0] = pos[..., 0] - top_left_px[0]
        pos[..., 1] = -pos[..., 1] - top_left_px[1]

        return np.round(pos).astype(np.int32)
    
    def _get_bounding_box_poly(self, state):
        x, y, yaw = state

        #agent state is measured from rear axle center, so adjust center
        #to be 
        heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
        # center = np.array(x, y) + heading*0.1
        # center_x = center[0]
        # center_y = center[1]

        rot = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)],
        ])

        lw = np.array([
            [-(self.agent_length / 2), -(self.agent_width / 2)],
            [+(self.agent_length / 2), -(self.agent_width / 2)],
            [+(self.agent_length / 2), +(self.agent_width / 2)],
            [-(self.agent_length / 2), +(self.agent_width / 2)],
        ])
        lw = np.matmul(lw, rot)

        bounding_box = np.array([x, y]) + (heading* 0.1) + lw
        return bounding_box

    def _draw_full_map(
        self, 
        ego, 
        agents, 
        map_objects, 
        collision_centroids = None,
        px_per_meter: float = 100.0, 
        edge_px_x: int = 200,
        edge_px_y: int = 600
    ):
        #switch y and x for rendering
        ymin, ymax, xmin, xmax = (self.map_boundary * px_per_meter).astype(np.int64)
        ymax *= -1
        ymin *= -1
        xmin -= edge_px_x
        ymin -= edge_px_y
        xmax += edge_px_x
        ymax += edge_px_y

        raster_map = np.zeros([xmax - xmin, ymax - ymin, 3], dtype=np.uint8)
        top_left_px = np.array([xmin, ymin - 200], dtype=np.float32)

        #render polylines
        for i in np.where(self.map_valid.any(axis=1))[0]:
            cv2.polylines(
                raster_map,
                [self._to_pixel(
                    self.map_pos[i][self.map_valid[i]], 
                    None, 
                    top_left_px=top_left_px, 
                    px_per_meter=px_per_meter)
                ],
                color=(255,255,255),
                thickness=1,
                isClosed=False,
                lineType=cv2.LINE_AA
            )

        ego_state = ego[:3]
        ego_box = self._get_bounding_box_poly(ego_state)

        #render agents
        #temp_raster = raster_map.copy() #np.zeros((192, 192, 3), dtype=np.uint8)
        agent_colors = {
            1: (62, 175, 252), #orange
            2: (252, 0, 255), #magenta
            3: (41, 41, 239), #scarlet
        }
        if agents is not None:
            for i, agent in enumerate(agents):
                agent_id = i + 1

                agent_state = agent[:3]
                agent_box = self._get_bounding_box_poly(agent_state)

                cv2.fillPoly(
                    raster_map,
                    [self._to_pixel(
                        agent_box, 
                        None,
                        top_left_px=top_left_px, 
                        px_per_meter=px_per_meter
                    )],
                    color=agent_colors[agent_id]
            )

        #render ego agent
        #temp_raster = np.zeros((192, 192, 3), dtype=np.uint8)
        cv2.fillPoly(
            raster_map,
            [self._to_pixel(
                ego_box, 
                None,
                top_left_px=top_left_px, 
                px_per_meter=px_per_meter
            )],
            color=(255, 0, 0)
        )

        #draw collision circles
        if collision_centroids is not None:
            for i in range(collision_centroids.shape[0]):
                for j in range(collision_centroids.shape[1]):
                    pos = collision_centroids[i, j]

                    cv2.circle(
                        raster_map,
                        self._to_pixel(
                            pos,
                            None,
                            top_left_px=top_left_px, 
                            px_per_meter=px_per_meter
                        ),
                        int(0.12 * px_per_meter),
                        (0, 255, 255),
                        -1
                    )

        cv2.imshow("RASTER", raster_map)
        cv2.waitKey(1)

        return raster_map
    
    #generate a random cycle from a given starting node
    def generate_random_cycle(self, start, min_length=3, max_length=15):
        #depth first search for finding all cycles that start and end at the starting point
        if start in self.removed_node_list:
            pass
        def dfs(start):
            fringe = [(start, [])]

            while fringe:
                node, path = fringe.pop()
                if path and node == start:
                    yield path
                    continue
                for next_edges in node.outEdges:
                    next_node = next_edges.toNode
                    if next_node in path:
                        continue
                    fringe.append((next_node, path + [next_node]))

        start_node = self.nodes[start]
        cycles = [[start_node] + path for path in dfs(start_node) if min_length <= len(path) <= max_length]
        num_cycles = len(cycles)

        return cycles[np.random.randint(num_cycles)]

    #generate a random cycle from a given starting node
    def generate_random_path(self, start, length=10):
        #depth first search for finding all cycles that start and end at the starting point
        def dfs(start):
            fringe = [(start, [])]

            while fringe:
                node, path = fringe.pop()
                if len(path) == length:
                    yield path
                    continue
                for next_edges in node.outEdges:
                    next_node = next_edges.toNode
                    if next_node in path:
                        continue
                    fringe.append((next_node, path + [next_node]))

        start_node = self.nodes[start]
        paths = [[start_node] + path for path in dfs(start_node)]
        num_paths = len(paths)

        try:
            out = paths[np.random.randint(num_paths)]
        except:
            breakpoint()

        return out

    #wrap as numpy array object
    def generate_path(self, sequence):
        if type(sequence) == np.ndarray:
            sequence = sequence.tolist()

        #convert from map node to index
        sequence_ids = sequence
        if not type(sequence_ids[0]) == int:
            sequence_ids = [node.index for node in sequence_ids]

        #generate path and find which waypoints belong to which road segment
        path = []
        segments = {}
        waypoint_index = 0
        for i in range(1, len(sequence_ids)):
            sub_sequence = sequence_ids[i-1:i+1]
            sub_sequence_path = np.array(super().generate_path(sub_sequence)).transpose(1, 0) #[N, (x, y)]
            sub_sequence_length = sub_sequence_path.shape[0]

            path.append(sub_sequence_path)
            segments[(waypoint_index, waypoint_index + sub_sequence_length - 1)] = np.array(sub_sequence)
            waypoint_index += sub_sequence_length

        path = np.vstack(path)
        return path, segments
    
    def find_shortest_path(self, startNode, goalNode):
        """Find the shortest path between two nodes using the A* algorithm.

        Args:
            startNode (int or RoadMapNode): Starting node (index or instance).
            goalNode (int or RoadMapNode): Goal node (index or instance).
            radius (float): Minimum turning radius.

        Returns:
            path: generated path as a 2xn numpy array
        """
        if startNode == goalNode:
            return None

        if type(startNode) == int:
            startNode = self.nodes[startNode]
        if type(goalNode) == int:
            goalNode = self.nodes[goalNode]

        # Initialize the open set and closed set
        openSet = []
        closedSet = set()

        # Add the start node to the open set with a cost of 0 and an
        # f-score equal to the heuristic estimate
        heapq.heappush(
            openSet,
            (0 + self._heuristic(startNode, goalNode), startNode)
        )

        # Initialize the g-scores for each node to infinity
        gScore = {node: float('inf') for node in self.nodes}
        gScore[startNode] = 0

        # Initialize the 'came from' (node, edge) pair
        # for each node to None
        cameFrom = {node: None for node in self.nodes}

        while openSet:
            # Pop the node with the lowest f-score from the open set
            currentNode = heapq.heappop(openSet)[1]

            if currentNode == goalNode:
                # Goal node found, construct the optimal path, then return
                path = goalNode.pose[:2,:]
                node = goalNode
                index_path = np.array([node.index])
                while True:
                    (node, edge) = cameFrom[node]
                    index_path = np.concatenate([index_path, [node.index]])
                    path = np.hstack((
                        node.pose[:2,:],
                        edge.waypoints,
                        path
                    ))
                    if cameFrom[node] is None:
                        break

                return path, index_path

            closedSet.add(currentNode)

            for edge in currentNode.outEdges:
                neighborNode = edge.toNode
                if neighborNode in closedSet:
                    # Neighbor node already explored
                    continue

                if edge.length is None:
                    tentative_g = float('inf')
                else:
                    tentative_g = gScore[currentNode] + edge.length

                if tentative_g < gScore[neighborNode]:
                    # New path to neighbor node found,
                    # update parent pointer and g-score
                    cameFrom[neighborNode] = (currentNode, edge)
                    gScore[neighborNode] = tentative_g

                    # Add the neighbor node to the open set with a cost
                    # equal to the g-score plus the heuristic estimate
                    hScore = self._heuristic(neighborNode, goalNode)
                    heapq.heappush(
                        openSet,
                        (gScore[neighborNode] + hScore, neighborNode)
                    )

        # Open set is empty and goal node not found, no path exists
        return None