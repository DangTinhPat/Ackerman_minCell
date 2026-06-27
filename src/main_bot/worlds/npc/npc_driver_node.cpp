// NPC driver node — spawns NPC box và điều khiển chạy theo oval track.
//
// Chiến lược chống drift + giữ collision:
//   • VelocityControl feedforward mỗi 20ms → physics engine có velocity thật
//     → robot KHÔNG xuyên qua NPC (contact forces hoạt động đúng)
//   • set_pose async mỗi CORRECT_INTERVAL bước (≈1s) → hiệu chỉnh drift tích lũy
//     → NPC không lệch khỏi lane theo thời gian
//   • Timer dùng sim time (create_timer) → đồng bộ với Gazebo clock

#include <cmath>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/entity_factory.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/transport/Node.hh>

using namespace std::chrono_literals;

// ── Track geometry ────────────────────────────────────────────────────────────
static constexpr double STRAIGHT_HALF = 6.0;
static constexpr double L_STRAIGHT    = 2.0 * STRAIGHT_HALF;   // 12.0 m
static constexpr double NPC_SPEED     = 0.25;                   // m/s

// ── NPC box dimensions ────────────────────────────────────────────────────────
// NPC_WID tăng 20% (0.158 → 0.190) để robot không dễ vượt qua và dễ phát hiện hơn
static constexpr double NPC_LEN = 0.297;
static constexpr double NPC_WID = 0.190;   // +20% so với 0.158
static constexpr double NPC_HGT = 0.167;
static constexpr double NPC_BODY_Z = NPC_HGT / 2.0 + 0.003;   // 0.0865 m

// ── Inertia (m=2.0 kg, cập nhật theo NPC_WID mới) ───────────────────────────
// ixx = m/12*(W²+H²) = 2/12*(0.190²+0.167²) = 0.01067
// iyy = m/12*(L²+H²) = 2/12*(0.297²+0.167²) = 0.01935  (không đổi)
// izz = m/12*(L²+W²) = 2/12*(0.297²+0.190²) = 0.02072
static const std::string NPC_INERTIA =
    "<ixx>0.01067</ixx><ixy>0</ixy><ixz>0</ixz>"
    "<iyy>0.01935</iyy><iyz>0</iyz>"
    "<izz>0.02072</izz>";

// ── Pose trên oval ────────────────────────────────────────────────────────────
struct Pose2D { double x, y, yaw; };

static Pose2D oval_pose(double s, double lane_y)
{
    const double lc = M_PI * lane_y;
    const double lt = 2.0 * L_STRAIGHT + 2.0 * lc;
    s = std::fmod(s, lt);
    if (s < 0.0) s += lt;

    if (s < L_STRAIGHT)
        return { -STRAIGHT_HALF + s, +lane_y, 0.0 };
    s -= L_STRAIGHT;

    if (s < lc) {
        double th = s / lane_y;
        return { STRAIGHT_HALF + lane_y * std::sin(th),
                 lane_y * std::cos(th), -th };
    }
    s -= lc;

    if (s < L_STRAIGHT)
        return { STRAIGHT_HALF - s, -lane_y, M_PI };
    s -= L_STRAIGHT;

    double th = s / lane_y;
    return { -(STRAIGHT_HALF + lane_y * std::sin(th)),
             -(lane_y * std::cos(th)), M_PI - th };
}

// ── SDF với VelocityControl (CẦN cho collision physics) ──────────────────────
static std::string make_sdf(const std::string & name)
{
    auto f = [](double v){ return std::to_string(v); };
    std::string sz = f(NPC_LEN) + " " + f(NPC_WID) + " " + f(NPC_HGT);
    return
        "<sdf version='1.6'>"
        "<model name='" + name + "'>"
        "<static>false</static>"

        "<link name='body'>"
          "<pose>0 0 " + f(NPC_BODY_Z) + " 0 0 0</pose>"
          "<gravity>false</gravity>"
          "<inertial>"
            "<mass>2.0</mass>"
            "<inertia>" + NPC_INERTIA + "</inertia>"
          "</inertial>"
          "<collision name='col'>"
            "<geometry><box><size>" + sz + "</size></box></geometry>"
            "<surface>"
              "<friction>"
                "<ode><mu>0.8</mu><mu2>0.8</mu2></ode>"
              "</friction>"
              "<contact>"
                "<ode><kp>1e5</kp><kd>10</kd><min_depth>0.001</min_depth></ode>"
              "</contact>"
            "</surface>"
          "</collision>"
          "<visual name='vis'>"
            "<geometry><box><size>" + sz + "</size></box></geometry>"
            "<material>"
              "<ambient>1.0 0.45 0.0 1</ambient>"
              "<diffuse>1.0 0.45 0.0 1</diffuse>"
              "<specular>0.15 0.10 0.0 1</specular>"
            "</material>"
          "</visual>"
        "</link>"

        // VelocityControl: cho NPC có velocity thật trong physics engine
        // → robot va chạm với NPC (contact forces hoạt động)
        "<plugin filename='gz-sim-velocity-control-system'"
                " name='gz::sim::systems::VelocityControl'>"
          "<topic>/model/" + name + "/cmd_vel</topic>"
        "</plugin>"

        "</model></sdf>";
}

// ── Node ──────────────────────────────────────────────────────────────────────
class NpcDriverNode : public rclcpp::Node
{
public:
    NpcDriverNode() : Node("npc_driver_node")
    {
        this->declare_parameter<std::string>("npc_name",    "npc_1");
        this->declare_parameter<double>     ("lane_y",      2.267);
        this->declare_parameter<double>     ("initial_arc", 0.0);

        npc_name_ = this->get_parameter("npc_name").as_string();
        lane_y_   = this->get_parameter("lane_y").as_double();
        arc_pos_  = this->get_parameter("initial_arc").as_double();

        RCLCPP_INFO(this->get_logger(),
            "[%s] lane_y=%.3f  initial_arc=%.1f  L_total=%.2f m  WID=%.3f m",
            npc_name_.c_str(), lane_y_, arc_pos_, l_total(), NPC_WID);

        spawn_timer_ = this->create_wall_timer(
            3s, std::bind(&NpcDriverNode::do_spawn, this));
    }

private:
    double l_curve() const { return M_PI * lane_y_; }
    double l_total() const { return 2.0 * L_STRAIGHT + 2.0 * l_curve(); }

    // Feedforward curvature → VelocityControl (giữ collision physics)
    void publish_cmd(double s)
    {
        const double lc = l_curve();
        const double lt = l_total();
        s = std::fmod(s, lt);
        if (s < 0.0) s += lt;

        double kappa = 0.0;
        if (s >= L_STRAIGHT && s < L_STRAIGHT + lc)
            kappa = -1.0 / lane_y_;   // right curve (CW)
        else if (s >= 2.0 * L_STRAIGHT + lc)
            kappa = -1.0 / lane_y_;   // left curve  (CW)

        gz::msgs::Twist cmd;
        cmd.mutable_linear()->set_x(NPC_SPEED);
        cmd.mutable_angular()->set_z(NPC_SPEED * kappa);
        cmd_pub_.Publish(cmd);
    }

    // Hiệu chỉnh pose qua service set_pose (async, không block) → chống drift
    void correct_pose(double s)
    {
        Pose2D p = oval_pose(s, lane_y_);

        gz::msgs::Pose msg;
        msg.set_name(npc_name_);
        msg.mutable_position()->set_x(p.x);
        msg.mutable_position()->set_y(p.y);
        msg.mutable_position()->set_z(0.0);
        msg.mutable_orientation()->set_w(std::cos(p.yaw / 2.0));
        msg.mutable_orientation()->set_x(0.0);
        msg.mutable_orientation()->set_y(0.0);
        msg.mutable_orientation()->set_z(std::sin(p.yaw / 2.0));

        gz_node_.Request<gz::msgs::Pose, gz::msgs::Boolean>(
            "/world/oval_lane_world/set_pose",
            msg,
            [](const gz::msgs::Boolean &, const bool) {});
    }

    void do_spawn()
    {
        spawn_timer_->cancel();

        Pose2D p0 = oval_pose(arc_pos_, lane_y_);

        gz::msgs::EntityFactory req;
        req.set_sdf(make_sdf(npc_name_));
        req.set_name(npc_name_);
        req.mutable_pose()->mutable_position()->set_x(p0.x);
        req.mutable_pose()->mutable_position()->set_y(p0.y);
        req.mutable_pose()->mutable_position()->set_z(0.0);
        req.mutable_pose()->mutable_orientation()->set_w(std::cos(p0.yaw / 2.0));
        req.mutable_pose()->mutable_orientation()->set_x(0.0);
        req.mutable_pose()->mutable_orientation()->set_y(0.0);
        req.mutable_pose()->mutable_orientation()->set_z(std::sin(p0.yaw / 2.0));

        gz::msgs::Boolean res;
        bool result = false;
        bool ok = gz_node_.Request(
            "/world/oval_lane_world/create", req, 5000, res, result);

        if (!ok || !result) {
            RCLCPP_WARN(this->get_logger(),
                "[%s] spawn failed, retry in 2 s", npc_name_.c_str());
            spawn_timer_ = this->create_wall_timer(
                2s, std::bind(&NpcDriverNode::do_spawn, this));
            return;
        }

        std::string topic = "/model/" + npc_name_ + "/cmd_vel";
        cmd_pub_ = gz_node_.Advertise<gz::msgs::Twist>(topic);

        RCLCPP_INFO(this->get_logger(),
            "[%s] spawned  pos=(%.2f, %.2f)  yaw=%.1f°"
            "  VelCtrl=%s  correct_every=%d steps",
            npc_name_.c_str(), p0.x, p0.y,
            p0.yaw * 180.0 / M_PI,
            topic.c_str(), CORRECT_INTERVAL);

        last_time_ = this->now();
        step_count_ = 0;

        // create_timer dùng sim clock (use_sim_time=true → đồng bộ Gazebo)
        move_timer_ = this->create_timer(
            std::chrono::milliseconds(20),
            std::bind(&NpcDriverNode::move_step, this));
    }

    void move_step()
    {
        auto now = this->now();
        double dt = (now - last_time_).seconds();
        last_time_ = now;

        arc_pos_ = std::fmod(arc_pos_ + NPC_SPEED * dt, l_total());
        step_count_++;

        // Mỗi bước: publish cmd_vel → VelocityControl giữ collision physics
        publish_cmd(arc_pos_);

        // Mỗi CORRECT_INTERVAL bước: hiệu chỉnh vị trí để chống drift lane
        if (step_count_ % CORRECT_INTERVAL == 0)
            correct_pose(arc_pos_);
    }

    // Số bước giữa 2 lần pose correction. 50 bước × 20ms = 1 giây.
    static constexpr int CORRECT_INTERVAL = 50;

    std::string  npc_name_;
    double       lane_y_{2.267};
    double       arc_pos_{0.0};
    int          step_count_{0};
    rclcpp::Time last_time_;

    gz::transport::Node            gz_node_;
    gz::transport::Node::Publisher cmd_pub_;

    rclcpp::TimerBase::SharedPtr spawn_timer_;
    rclcpp::TimerBase::SharedPtr move_timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NpcDriverNode>());
    rclcpp::shutdown();
    return 0;
}
