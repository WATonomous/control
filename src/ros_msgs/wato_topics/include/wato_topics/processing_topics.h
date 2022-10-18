#pragma once

#include <string>

namespace processing_topics
{
static const std::string PROCESSING_TOPICS = "processing/";

static const std::string LANE_POST = PROCESSING_TOPICS + "roadline_post_processing";
static const std::string STOP_DETECT = PROCESSING_TOPICS + "stopline_detection_output";
static const std::string ENV_OUTPUT = PROCESSING_TOPICS + "environment_prediction";
static const std::string HLDF_OUTPUT = PROCESSING_TOPICS + "environment_hldf";
static const std::string HD_MAP = PROCESSING_TOPICS + "hd_map";
} // namespace processing_topics
