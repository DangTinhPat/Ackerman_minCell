#pragma once
#include <rclcpp/time.hpp>

enum class OvertakeState { FOLLOW, PREPARE, OVERTAKE, RETURN };

class StateMachine
{
public:
    struct Config {
        double prepare_hold_time{1.0};
        double overtake_hold_time{6.0};
        double return_tol{0.04};
    };

    // Cập nhật state machine; trả true nếu state vừa thay đổi.
    // can_overtake      : front_present && adj_clear && gap_ok  → trigger FOLLOW→PREPARE
    // can_prepare_hold  : front_present && adj_clear             → giữ trong PREPARE
    //   (không cần gap_ok để giữ PREPARE — tránh oscillation khi NPC angle thay đổi)
    // right_dist        : min sector phải (-135° đến -45°) — dùng kiểm tra OVERTAKE→RETURN
    // cur_offset        : target_offset hiện tại            — dùng kiểm tra RETURN→FOLLOW
    bool update(bool can_overtake, bool can_prepare_hold, bool abort,
                double right_dist, double cur_offset,
                const Config & cfg, const rclcpp::Time & now);

    OvertakeState state()      const { return state_; }
    const char *  state_name() const;

private:
    OvertakeState state_{OvertakeState::FOLLOW};
    // PHẢI init với RCL_ROS_TIME vì node dùng use_sim_time=true.
    // rclcpp::Time default = RCL_SYSTEM_TIME ≠ RCL_ROS_TIME → operator- throws → state machine chết.
    rclcpp::Time  state_entry_time_{0, 0, RCL_ROS_TIME};

    void   transition_to(OvertakeState s, const rclcpp::Time & now);
    double time_in_state(const rclcpp::Time & now) const;
};
