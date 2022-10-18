#pragma once

#include <string>

namespace path_planning_topics {
static const std::string PATH_PLANNING_TOPICS = "path_planning/";

static const std::string ENVIRONMENT_TOPIC = PATH_PLANNING_TOPICS + "environment";
static const std::string PATH_TOPIC = PATH_PLANNING_TOPICS + "path";
static const std::string CAN_TOPIC = PATH_PLANNING_TOPICS + "feedback_desired_output";
} // path_planning_topics
