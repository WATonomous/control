#pragma once

#include <string>

namespace perception_topics {
static const std::string PERCEPTION_TOPICS = "/"; // Update to perception/ later

static const std::string ROAD_LINE_TOPIC = PERCEPTION_TOPICS + "roadline_post_processing";
static const std::string TRAFFIC_LIGHT_TOPIC = PERCEPTION_TOPICS + "traffic_light_detection";
static const std::string TRAFFIC_SIGN_TOPIC = PERCEPTION_TOPICS + "traffic_sign_detection";
static const std::string LIDAR_DETECTION_TOPIC = PERCEPTION_TOPICS + "object_detection";
static const std::string OBSTACLE_TOPIC = PERCEPTION_TOPICS + "obstacle_detection";

// internal topics
static const std::string JSK_LIDAR_BOXES_TOPIC = PERCEPTION_TOPICS + "jsk_bboxes";
static const std::string MERGED_LIDAR_TOPIC = PERCEPTION_TOPICS + "lidar_merged_visualized";
static const std::string OBSTACLE_DEBUG_TOPIC = PERCEPTION_TOPICS + "obstacle_detection_visualizer";
static const std::string ROAD_LINE_DEBUG_TOPIC = PERCEPTION_TOPICS + "roadline_detection_visualizer";
static const std::string ROAD_LINE_MASK_DEBUG_TOPIC = PERCEPTION_TOPICS + "raw_mask_roadline_detection_visualizer";
static const std::string TRAFFIC_LIGHT_DEBUG_TOPIC = PERCEPTION_TOPICS + "traffic_light_detection_visualizer";
static const std::string TRAFFIC_SIGN_DEBUG_TOPIC = PERCEPTION_TOPICS + "traffic_sign_detection_visualizer";
static const std::string ROAD_LINE_MASK_TOPIC = PERCEPTION_TOPICS + "roadline_detections"; 
static const std::string TRAFFIC_LIGHT_INTERNAL_TOPIC = PERCEPTION_TOPICS + "traffic_light_internal"; 
} // perception_topics
