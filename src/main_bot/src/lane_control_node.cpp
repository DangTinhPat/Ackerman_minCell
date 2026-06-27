#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float64.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

// ── Sơ đồ điều khiển (Stanley + Feed-forward curvature + Odometry kappa) ────
//
//  /odometry/filtered ──→ κ_odom = vyaw / vx   [curvature thực tế, 50 Hz, từ EKF]
//  /status_err        ──→ κ_vis                [curvature vision, 10 Hz, nhìn trước]
//                                │
//            κ_use = kappa_blend·κ_odom + (1−kappa_blend)·κ_vis
//              (khi vx > kappa_odom_min_vx; fallback κ_vis khi tốc độ thấp)
//                                │
//  κ_use ──→ delta_ff = k_ff · atan(L · κ_use)    [bù góc cong hình học]
//  e_psi ──→ [dead-band] ──→ e_psi_corr = e_psi − X_avg·κ_use
//  e_y   ──→ [dead-band] ──→ atan2(k·e_y, v_soft)
//                                          │
//                    delta = delta_ff + e_psi_corr + atan2(k·e_y, v_soft)
//                                          │  clamp ±max_steer
//                              raw_az = v·tan(delta) / L
//                                          │
//                   [Asymmetric EMA: α_normal khi tăng, α_conv khi giảm]
//                                          │
//                                 angular_z → /cmd_vel
//
// Tại sao blend κ_odom + κ_vis:
//   κ_vis  : predictive (nhìn trước ~0.9m) nhưng nhiễu từ RANSAC
//   κ_odom : smooth từ EKF gyro/encoder nhưng phản ánh cua hiện tại (không trước)
//   blend=0.6 → 60% odom (ổn định) + 40% vis (predictive) = cân bằng tốt nhất

class LaneControlNode : public rclcpp::Node
{
public:
  LaneControlNode() : Node("lane_control_node")
  {
    this->declare_parameter("speed",        8.0);
    this->declare_parameter("k",            0.30);
    this->declare_parameter("max_steer",    0.52);
    this->declare_parameter("timeout",      0.5);
    this->declare_parameter("v_softening",  0.20);
    this->declare_parameter("alpha",        0.40);
    // alpha_conv: EMA alpha khi |raw_az| < |angular_z_smooth| (đang giảm tốc / thoát cua)
    // Lớn hơn alpha thường → output hội tụ về 0 nhanh → tắt dao động sau đường cong
    // α=0.70 → τ≈125ms để decay 5% vs α=0.40 → τ≈295ms
    this->declare_parameter("alpha_conv",   0.70);
    // Dead-band
    this->declare_parameter("db_ey",        0.020);
    this->declare_parameter("db_psi",       0.017);
    // Feed-forward curvature:
    //   k_ff      : gain cho delta_ff = atan(L·κ).  1.0 = bù hoàn toàn.
    //   lookahead_x: X_avg của 2 điểm near/far trong world [m].
    //                = (X_near + X_far)/2 = (0.665 + 1.181)/2 = 0.923m
    //                Dùng để tách curvature ra khỏi e_psi.
    this->declare_parameter("k_ff",         1.00);
    this->declare_parameter("lookahead_x",  0.923);
    // κ từ odometry:
    //   kappa_blend      : trọng số κ_odom trong blend [0=all-vision, 1=all-odom]
    //                      0.6 → 60% odom (ổn định) + 40% vision (predictive)
    //   kappa_odom_min_vx: tốc độ tối thiểu để κ_odom = vyaw/vx đáng tin [m/s]
    //                      dưới ngưỡng này fallback κ_vision tránh chia cho 0
    this->declare_parameter("kappa_blend",       0.60);
    this->declare_parameter("kappa_odom_min_vx", 0.05);

    speed_        = this->get_parameter("speed").as_double();
    k_            = this->get_parameter("k").as_double();
    max_steer_    = this->get_parameter("max_steer").as_double();
    timeout_      = this->get_parameter("timeout").as_double();
    v_softening_  = this->get_parameter("v_softening").as_double();
    alpha_        = this->get_parameter("alpha").as_double();
    alpha_conv_   = this->get_parameter("alpha_conv").as_double();
    db_ey_        = this->get_parameter("db_ey").as_double();
    db_psi_       = this->get_parameter("db_psi").as_double();
    k_ff_             = this->get_parameter("k_ff").as_double();
    lookahead_x_      = this->get_parameter("lookahead_x").as_double();
    kappa_blend_      = this->get_parameter("kappa_blend").as_double();
    kappa_odom_min_vx_ = this->get_parameter("kappa_odom_min_vx").as_double();

    sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
      "/status_err", 10,
      std::bind(&LaneControlNode::err_callback, this, std::placeholders::_1));

    sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odometry/filtered", 10,
      std::bind(&LaneControlNode::odom_callback, this, std::placeholders::_1));

    // Overtake planner: nhận target_offset và áp vào e_y
    sub_offset_ = this->create_subscription<std_msgs::msg::Float64>(
      "/overtake/target_offset", 10,
      [this](std_msgs::msg::Float64::SharedPtr msg) { target_offset_ = msg->data; });

    // Overtake speed control: giảm tốc khi tiếp cận NPC, tăng khi vượt
    sub_speed_ = this->create_subscription<std_msgs::msg::Float64>(
      "/overtake/target_speed", 10,
      [this](std_msgs::msg::Float64::SharedPtr msg) {
          if (msg->data > 0.0) speed_ = msg->data;
      });

    pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    timer_ = this->create_wall_timer(
      50ms, std::bind(&LaneControlNode::control_loop, this));

    RCLCPP_INFO(this->get_logger(),
      "Stanley+FF+κ-blend: speed=%.2f  k=%.2f  v_soft=%.2f  "
      "alpha=%.2f/%.2f  db_ey=%.3f  db_psi=%.1fdeg  "
      "k_ff=%.2f  lookahead_x=%.3fm  kappa_blend=%.2f(odom)/%.2f(vis)",
      speed_, k_, v_softening_,
      alpha_, alpha_conv_, db_ey_, db_psi_ * 180.0 / M_PI,
      k_ff_, lookahead_x_, kappa_blend_, 1.0 - kappa_blend_);
  }

private:
  // ── Stanley + Output EMA ─────────────────────────────────────────────────
  //   delta     = e_psi + atan(k * e_y / max(v, v_softening))
  //   raw_az    = v * tan(delta) / L
  //   angular_z = alpha * raw_az + (1-alpha) * angular_z_prev   ← output EMA duy nhất

  static constexpr double WHEELBASE_M = 0.21;

  void err_callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
  {
    e_y_           = msg->x;
    e_psi_         = msg->y;
    kappa_         = msg->z;
    last_err_time_ = this->now();
    has_received_  = true;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double vx   = msg->twist.twist.linear.x;
    double vyaw = msg->twist.twist.angular.z;
    if (vx > kappa_odom_min_vx_) {
      // κ = vyaw / vx : curvature tức thời, smooth từ EKF 50 Hz
      // clip ±2.0 tránh spike khi vx vừa vượt ngưỡng
      kappa_odom_ = std::clamp(vyaw / vx, -2.0, 2.0);
      odom_valid_ = true;
    } else {
      odom_valid_ = false;   // tốc độ thấp, κ_odom không đáng tin
    }
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

    // Dead-band: bỏ qua nhiễu nhỏ, giữ nguyên sai số lớn
    auto deadband = [](double x, double db) {
      return std::abs(x) < db ? 0.0 : x;
    };
    // Áp target_offset từ overtake planner trước khi tính dead-band
    // effective_e_y = e_y - target_offset:
    //   target_offset = -0.534 → effective = e_y + 0.534 → lái sang làn ngoài
    //   target_offset = 0.0   → bám lane gốc bình thường
    double e_y_in   = deadband(e_y_ - target_offset_, db_ey_);
    double e_psi_in = deadband(e_psi_, db_psi_);

    // ── Tổng hợp κ: blend κ_odom (EKF, smooth) + κ_vision (predictive) ──────
    // κ_odom = vyaw/vx từ /odometry/filtered — 50 Hz, ổn định, phản ánh cua HIỆN TẠI
    // κ_vis  = từ camera RANSAC — 10 Hz, nhìn trước ~0.9m, nhưng nhiễu hơn
    // Khi odom_valid: blend có trọng số → giảm nhiễu vision, giữ tính predictive
    // Khi !odom_valid (vx thấp): dùng κ_vis hoàn toàn
    double kappa_use;
    if (odom_valid_) {
      kappa_use = kappa_blend_ * kappa_odom_ + (1.0 - kappa_blend_) * kappa_;
    } else {
      kappa_use = kappa_;
    }

    // Feed-forward: bù góc cong hình học với κ đã tổng hợp
    //   delta_ff   = atan(L · κ)       — lái đúng bám cung tròn
    //   e_psi_corr = e_psi − X_avg·κ  — chỉ còn alignment error, không còn curvature
    double delta_ff    = k_ff_ * std::atan(WHEELBASE_M * kappa_use);
    double e_psi_corr  = e_psi_in - lookahead_x_ * kappa_use;

    double v_denom = std::max(v, v_softening_);
    double delta   = delta_ff + e_psi_corr + std::atan2(k_ * e_y_in, v_denom);
    delta          = std::clamp(delta, -max_steer_, max_steer_);

    double raw_az = v * std::tan(delta) / WHEELBASE_M;

    // Asymmetric EMA: tắt dao động sau đường cong bằng cách hội tụ nhanh hơn
    // khi lệnh output đang giảm (thoát cua / thẳng lại).
    //   |raw_az| <  |smooth|  → đang giảm tốc → dùng alpha_conv (nhanh)
    //   |raw_az| >= |smooth|  → đang tăng tốc → dùng alpha     (chậm, mượt)
    bool  decelerating = std::abs(raw_az) < std::abs(angular_z_smooth_);
    double alpha_eff   = decelerating ? alpha_conv_ : alpha_;
    angular_z_smooth_  = alpha_eff * raw_az + (1.0 - alpha_eff) * angular_z_smooth_;

    twist.linear.x  = v;
    twist.angular.z = angular_z_smooth_;
    pub_->publish(twist);
  }

  // ── Parameters ───────────────────────────────────────────────────────────
  double speed_{8.0};
  double k_{0.30};
  double max_steer_{0.52};
  double timeout_{0.5};
  double v_softening_{0.20};
  double alpha_{0.40};
  double alpha_conv_{0.70};
  double db_ey_{0.020};
  double db_psi_{0.017};
  double k_ff_{1.00};
  double lookahead_x_{0.923};
  double kappa_blend_{0.60};
  double kappa_odom_min_vx_{0.05};

  // ── State ────────────────────────────────────────────────────────────────
  double       e_y_{0.0};
  double       e_psi_{0.0};
  double       kappa_{0.0};      // κ từ vision
  double       kappa_odom_{0.0}; // κ từ odometry (vyaw/vx)
  bool         odom_valid_{false};
  double       angular_z_smooth_{0.0};
  rclcpp::Time last_err_time_;
  bool         has_received_{false};
  double       target_offset_{0.0};  // từ overtake_node, default 0 (không dịch)

  // ── ROS interfaces ───────────────────────────────────────────────────────
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr   sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       sub_odom_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr        sub_offset_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr        sub_speed_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr        pub_;
  rclcpp::TimerBase::SharedPtr                                    timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaneControlNode>());
  rclcpp::shutdown();
  return 0;
}
