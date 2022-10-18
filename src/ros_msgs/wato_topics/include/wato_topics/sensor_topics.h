#pragma once

#include <string>

namespace sensor_topics {
static const std::string SENSOR_TOPICS = "sensor/";
static const std::string CAMERA_TOPICS = "camera/";
static const std::string CAMERA_RFL_RAW = SENSOR_TOPICS + CAMERA_TOPICS + "rfl_raw";
static const std::string CAMERA_RFR_RAW = SENSOR_TOPICS + CAMERA_TOPICS + "rfr_raw";
static const std::string CAMERA_BC_RAW = SENSOR_TOPICS + CAMERA_TOPICS + "bc_raw";

static const std::string NAV_TOPICS = "navsat/";
static const std::string ODOM_TOPIC = NAV_TOPICS + "odom";
} // sensor_topics
