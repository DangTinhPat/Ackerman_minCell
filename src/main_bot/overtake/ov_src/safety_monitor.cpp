#include "../ov_inc/safety_monitor.hpp"
#include <algorithm>
#include <cmath>

void SafetyMonitor::on_scan(sensor_msgs::msg::LaserScan::SharedPtr msg) { latest_scan_ = msg; }

void SafetyMonitor::on_odom(nav_msgs::msg::Odometry::SharedPtr msg)
{
    double vx   = msg->twist.twist.linear.x;
    double vyaw = msg->twist.twist.angular.z;
    v_ego_ = std::abs(vx);
    // kappa = vyaw/vx (curvature), valid chỉ khi xe đang chạy đủ nhanh
    if (std::abs(vx) > 0.05)
        kappa_ = std::clamp(vyaw / vx, -2.0, 2.0);
}

void SafetyMonitor::on_imu(sensor_msgs::msg::Imu::SharedPtr msg)
{
    a_lateral_ = msg->linear_acceleration.y;
}

double SafetyMonitor::min_sector(double a_from, double a_to) const
{
    if (!latest_scan_) return 999.0;
    const auto & s = *latest_scan_;
    int n = static_cast<int>(s.ranges.size());
    if (n == 0 || s.angle_increment <= 0) return 999.0;
    if (a_from > a_to) std::swap(a_from, a_to);
    // 999.0 làm sentinel "không có vật cản" — tránh nhầm range_max là vật cản thật
    // khi front_detect_range > lidar_max (e.g. 5.0m detect range với 4.0m lidar)
    double res = 999.0;
    int i0 = std::clamp((int)((a_from - s.angle_min) / s.angle_increment), 0, n - 1);
    int i1 = std::clamp((int)((a_to   - s.angle_min) / s.angle_increment), 0, n - 1);
    for (int i = i0; i <= i1; ++i) {
        float r = s.ranges[i];
        if (std::isfinite(r) && r > s.range_min && r < s.range_max)
            res = std::min(res, static_cast<double>(r));
    }
    return res;
}

SafetyResult SafetyMonitor::run(const Config & cfg)
{
    SafetyResult out;

    // ── Curve compensation ─────────────────────────────────────────────────────
    // Trên đường cong bán kính R, NPC cách D phía trước xuất hiện ở góc θ = D·κ.
    // Oval track: R_inner ≈ 3 m → κ_max ≈ 0.33. Với D=3m → θ ≈ 57° > 30° front sector!
    // → Dịch tâm front sector về hướng NPC thực tế.
    //
    // theta_bias > 0 = quay trái (kappa > 0), NPC lệch sang trái trong LiDAR.
    // Dùng D_mid = giữa khoảng phát hiện để tính bias trung bình.
    const double D_mid      = (cfg.front_safe_min + cfg.front_detect_range) * 0.5;
    const double max_bias   = 55.0 * M_PI / 180.0;   // giới hạn ±55°
    const double theta_bias = std::clamp(D_mid * kappa_, -max_bias, max_bias);
    out.theta_bias = theta_bias;

    // Module 1: Front sector — union của biased (trên curve) và unbiased (thẳng)
    // Vấn đề: theta_bias=45° → sector [15°,75°] → NPC thẳng phía trước (0°) bị bỏ sót!
    // Giải pháp: lấy MIN của 2 sector để bắt được NPC dù ở thẳng hay cong
    //   unbiased [-30°, +30°]   : bắt NPC thẳng trước mặt
    //   biased   [bias-30°, bias+30°]: bắt NPC trên đường cong
    const double half_rad = cfg.front_sector_deg * M_PI / 180.0;
    front_dist_ = std::min(
        min_sector(-half_rad, half_rad),
        min_sector(theta_bias - half_rad, theta_bias + half_rad)
    );
    bool front_ok = (front_dist_ > cfg.front_safe_min) && (front_dist_ < cfg.front_detect_range);
    out.front_dist    = front_dist_;
    out.front_present = front_ok;

    // Module 2: Adj sector (làn ngoài — bên trái robot)
    // Trên đường cong trái (theta_bias > 0): xe làn ngoài xuất hiện về phía trước hơn
    // → adj_lo co lại về phía forward để bắt đúng.
    // Trên đường cong phải (theta_bias < 0): xe làn ngoài lui về phía sau hơn
    // → adj_hi mở rộng về phía sau một chút.
    //
    // Công thức: xoay toàn bộ cửa sổ adj theo theta_bias (cùng chiều với front).
    const double deg2r = M_PI / 180.0;
    double adj_lo = std::clamp(45.0 * deg2r - theta_bias,
                                10.0 * deg2r, 70.0 * deg2r);
    double adj_hi = std::clamp(135.0 * deg2r - theta_bias * 0.5,
                                90.0 * deg2r, 165.0 * deg2r);
    double adj_dist = min_sector(adj_lo, adj_hi);
    out.adj_clear = (adj_dist > cfg.adjacent_clear_min);

    // Module 3: Gap time  —  v_rel = v_ego - v_npc (>0 = đang áp sát)
    double v_rel = v_ego_ - cfg.npc_speed;
    if (front_ok && v_rel > 0.05)
        out.gap_ok = (front_dist_ / v_rel) < cfg.gap_time_threshold;

    // Module 4: Same-lane distance — lateral filter để lọc xe làn bên cạnh
    // Với mỗi tia LiDAR trong ±60°, chỉ tính tia có lateral_offset < same_lane_half_width.
    // lateral_offset = r * |sin(angle - theta_bias)|
    // NPC outer lane tại 90°, cách 0.534m → lateral=0.534m >> 0.15m → loại bỏ ✓
    // NPC cùng làn tại 2°, cách 2.0m     → lateral=0.07m  <  0.15m → tính vào ✓
    if (latest_scan_) {
        const auto & s = *latest_scan_;
        int    n      = static_cast<int>(s.ranges.size());
        double sl_dist = 999.0;  // sentinel — tránh nhầm range_max là vật cản cùng làn
        // Union sector: [-60°, max(60°, bias+60°)] để cover cả thẳng và cong
        double sl_lo = std::min(theta_bias - 60.0 * M_PI / 180.0, -60.0 * M_PI / 180.0);
        double sl_hi = std::max(theta_bias + 60.0 * M_PI / 180.0,  60.0 * M_PI / 180.0);
        int sl_i0 = std::clamp((int)((sl_lo - s.angle_min) / s.angle_increment), 0, n - 1);
        int sl_i1 = std::clamp((int)((sl_hi - s.angle_min) / s.angle_increment), 0, n - 1);
        for (int i = sl_i0; i <= sl_i1; ++i) {
            float r = s.ranges[i];
            if (!std::isfinite(r) || r <= s.range_min || r >= s.range_max) continue;
            double angle = s.angle_min + i * s.angle_increment;
            // Lateral MIN của biased và unbiased — NPC thẳng trước (angle≈0°) bị lọc sai
            // khi dùng sin(0° - 45°)=0.71 >> 0.15m; unbiased sin(0°)=0 < 0.15 → đúng
            double lateral = std::min(
                std::abs(r * std::sin(angle - theta_bias)),
                std::abs(r * std::sin(angle))
            );
            if (lateral < cfg.same_lane_half_width)
                sl_dist = std::min(sl_dist, static_cast<double>(r));
        }
        out.same_lane_dist = sl_dist;
    }

    return out;
}

bool SafetyMonitor::check_abort(const Config & cfg, double dt,
                                 double prev_front_dist, bool in_overtake,
                                 double overtake_offset,
                                 std::string & reason) const
{
    // Abort 1: IMU gia tốc ngang (Layer 6)
    if (std::abs(a_lateral_) > cfg.imu_ay_limit) {
        reason = "IMU a_y=" + std::to_string(a_lateral_) + " m/s²";
        return true;
    }

    // Abort 2: vật cản phía trước
    // Khi đang OVERTAKE và robot đã dịch > 0.25m sang ngoài, NPC inner lane là expected
    // → chỉ abort nếu sắp va chạm thực sự (< 0.05m) chứ không dùng abort_front_dist
    bool expect_inner_npc = in_overtake && (std::abs(overtake_offset) > 0.25);
    double front_abort_thr = expect_inner_npc ? 0.05 : cfg.abort_front_dist;
    if (front_dist_ < front_abort_thr) {
        reason = "front=" + std::to_string(front_dist_) + "m thr=" + std::to_string(front_abort_thr);
        return true;
    }

    if (in_overtake) {
        // Abort 3: NPC phanh gấp (closing nhanh hơn dự kiến 0.5 m/s)
        double closing_rate = (prev_front_dist - front_dist_) / std::max(dt, 0.01);
        double expected     = v_ego_ - cfg.npc_speed;
        if (closing_rate > expected + 0.5) {
            reason = "NPC braking closing=" + std::to_string(closing_rate) + " m/s";
            return true;
        }

        // Abort 4: vật thể ở làn đang chuyển sang — sector trái 25°–160° (curve-aware)
        // theta_bias đã tính trong run() — dùng kappa_ trực tiếp
        const double D_mid    = 2.0;
        const double tb       = std::clamp(D_mid * kappa_, -55.0 * M_PI / 180.0,
                                                            55.0 * M_PI / 180.0);
        // Sector 25°–120° (không đến 160° vì tường outer lane ở ~90° cách 0.267m)
        // Ngưỡng 0.18m = robot half-width(0.079) + 0.10m → chặn đúng va chạm thật
        // (outer lane center → wall = 0.267m > 0.18m → không abort khi đến đích ✓)
        double left_close = min_sector(25.0 * M_PI / 180.0 - tb * 0.5,
                                       120.0 * M_PI / 180.0);
        if (left_close < 0.18) {
            reason = "oncoming left=" + std::to_string(left_close) + " m";
            return true;
        }
    }

    return false;
}
