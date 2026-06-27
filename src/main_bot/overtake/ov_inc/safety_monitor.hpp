#pragma once
#include <memory>
#include <string>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

struct SafetyResult {
    double front_dist{999.0};
    bool   front_present{false};
    bool   adj_clear{true};
    bool   gap_ok{false};
    double theta_bias{0.0};      // curvature bias rad — cho overtake_node dùng right_dist
    double same_lane_dist{999.0}; // khoảng cách chướng ngại vật CÙNG LÀN — dùng giảm tốc
};

class SafetyMonitor
{
public:
    struct Config {
        double front_detect_range{3.0};
        double front_safe_min{0.40};
        double front_sector_deg{30.0};    // half-angle của front sector (±deg)
        double adjacent_clear_min{0.50};
        double npc_speed{0.25};
        double gap_time_threshold{4.0};
        double abort_front_dist{0.35};
        double imu_ay_limit{3.0};
        // Same-lane filter (robot-proportional):
        //   track_width/2 = 0.108m, chassis_width/2 = 0.079m
        //   same_lane_half_width = track_width/2 + 0.04m safety margin ≈ 0.15m
        double same_lane_half_width{0.15};
    };

    void on_scan(sensor_msgs::msg::LaserScan::SharedPtr msg);
    void on_odom(nav_msgs::msg::Odometry::SharedPtr msg);
    void on_imu(sensor_msgs::msg::Imu::SharedPtr msg);

    // Layer 2: chạy Module 1-3, trả kết quả
    SafetyResult run(const Config & cfg);

    // Layer 6-7: kiểm tra abort; reason được điền nếu trả true
    // overtake_offset: target_offset hiện tại từ OffsetPlanner (0=inner, -0.534=outer)
    bool check_abort(const Config & cfg, double dt,
                     double prev_front_dist, bool in_overtake,
                     double overtake_offset,
                     std::string & reason) const;

    // Public để overtake_node lấy sector tuỳ chỉnh
    double min_sector(double a_from, double a_to) const;

    double front_dist() const { return front_dist_; }
    double v_ego()      const { return v_ego_; }
    double kappa()      const { return kappa_; }

private:
    sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;
    double v_ego_{0.0};
    double a_lateral_{0.0};
    double front_dist_{999.0};
    double kappa_{0.0};   // curvature vyaw/vx [rad/m], +ve = left curve
};
