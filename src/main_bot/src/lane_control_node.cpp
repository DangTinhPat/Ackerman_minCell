#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class LaneControlNode : public rclcpp::Node
{
public:
  LaneControlNode() : Node("lane_control_node")
  {
    this->declare_parameter("speed",        0.15);
    this->declare_parameter("k",            0.6);
    this->declare_parameter("max_steer",    0.52);
    this->declare_parameter("timeout",      0.5);
    // v_softening: tốc độ tối thiểu dùng trong mẫu số Stanley
    // Tránh lái quá gắt khi tốc độ thực thấp (v nhỏ → k*e_y/v rất lớn)
    this->declare_parameter("v_softening",  0.30);
    // alpha: hệ số EMA làm mịn angular_z (0=đứng yên, 1=không lọc)
    this->declare_parameter("alpha",        0.35);

    speed_       = this->get_parameter("speed").as_double();
    k_           = this->get_parameter("k").as_double();
    max_steer_   = this->get_parameter("max_steer").as_double();
    timeout_     = this->get_parameter("timeout").as_double();
    v_softening_ = this->get_parameter("v_softening").as_double();
    alpha_       = this->get_parameter("alpha").as_double();

    sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
      "/status_err", 10,
      std::bind(&LaneControlNode::err_callback, this, std::placeholders::_1));

    pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    timer_ = this->create_wall_timer(
      50ms, std::bind(&LaneControlNode::control_loop, this));

    RCLCPP_INFO(this->get_logger(),
      "Stanley lane control: speed=%.2f m/s  k=%.2f  "
      "max_steer=%.1f deg  v_soft=%.2f  alpha=%.2f  timeout=%.1f s",
      speed_, k_, max_steer_ * 180.0 / M_PI, v_softening_, alpha_, timeout_);
  }

private:
  // ── Stanley controller với EMA smoothing ────────────────────────────────
  //   delta     = e_psi + atan(k * e_y / max(v, v_softening))
  //   raw_az    = v * tan(delta) / L
  //   angular_z = alpha * raw_az + (1-alpha) * angular_z_prev   ← EMA

  static constexpr double WHEELBASE_M = 0.21;

  void err_callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
  {
    e_y_           = msg->x;
    e_psi_         = msg->y;
    last_err_time_ = this->now();
    has_received_  = true;
  }

  void control_loop()
  {
    geometry_msgs::msg::Twist twist;

    if (!has_received_) {
      angular_z_smooth_ = 0.0;
      pub_->publish(twist);
      return;
    }

    double elapsed = (this->now() - last_err_time_).seconds();
    if (elapsed > timeout_) {
      angular_z_smooth_ = 0.0;
      pub_->publish(twist);  // safety stop
      return;
    }

    double v = speed_;

    // v_softening ngăn mẫu số quá nhỏ khi tốc độ thực thấp
    double v_denom = std::max(v, v_softening_);
    double delta   = e_psi_ + std::atan2(k_ * e_y_, v_denom);
    delta          = std::clamp(delta, -max_steer_, max_steer_);

    double raw_az = v * std::tan(delta) / WHEELBASE_M;

    // EMA: làm mịn lệnh lái, chống giật khi vision nhiễu
    angular_z_smooth_ = alpha_ * raw_az + (1.0 - alpha_) * angular_z_smooth_;

    twist.linear.x  = v;
    twist.angular.z = angular_z_smooth_;
    pub_->publish(twist);
  }

  // ── Parameters ───────────────────────────────────────────────────────────
  double speed_{0.15};
  double k_{0.6};
  double max_steer_{0.52};
  double timeout_{0.5};
  double v_softening_{0.30};
  double alpha_{0.35};

  // ── State ────────────────────────────────────────────────────────────────
  double       e_y_{0.0};
  double       e_psi_{0.0};
  double       angular_z_smooth_{0.0};
  rclcpp::Time last_err_time_;
  bool         has_received_{false};

  // ── ROS interfaces ───────────────────────────────────────────────────────
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr      pub_;
  rclcpp::TimerBase::SharedPtr                                  timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaneControlNode>());
  rclcpp::shutdown();
  return 0;
}