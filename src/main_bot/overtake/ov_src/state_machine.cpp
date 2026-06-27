#include "../ov_inc/state_machine.hpp"
#include <cmath>

const char * StateMachine::state_name() const
{
    static const char * names[] = {"FOLLOW", "PREPARE", "OVERTAKE", "RETURN"};
    return names[static_cast<int>(state_)];
}

void StateMachine::transition_to(OvertakeState s, const rclcpp::Time & now)
{
    state_            = s;
    state_entry_time_ = now;
}

double StateMachine::time_in_state(const rclcpp::Time & now) const
{
    return (now - state_entry_time_).seconds();
}

bool StateMachine::update(bool can_overtake, bool can_prepare_hold, bool abort,
                           double right_dist, double cur_offset,
                           const Config & cfg, const rclcpp::Time & now)
{
    OvertakeState prev = state_;

    if (abort) {
        if (state_ != OvertakeState::FOLLOW)
            transition_to(OvertakeState::FOLLOW, now);
        return state_ != prev;
    }

    double t = time_in_state(now);

    switch (state_) {
        case OvertakeState::FOLLOW:
            if (can_overtake)
                transition_to(OvertakeState::PREPARE, now);
            break;

        case OvertakeState::PREPARE:
            // Chỉ huỷ PREPARE khi NPC biến mất hoàn toàn hoặc làn trái bị chặn.
            // Không dùng gap_ok — tránh oscillation khi góc NPC thay đổi trong curve.
            if (!can_prepare_hold)
                transition_to(OvertakeState::FOLLOW, now);
            else if (t >= cfg.prepare_hold_time)
                transition_to(OvertakeState::OVERTAKE, now);
            break;

        case OvertakeState::OVERTAKE:
            // Chuyển sang RETURN sau khi ở đủ lâu VÀ NPC đã ở phía sau phải
            if (t >= cfg.overtake_hold_time && right_dist > 1.0)
                transition_to(OvertakeState::RETURN, now);
            break;

        case OvertakeState::RETURN:
            // Offset đã hội tụ về 0 → về FOLLOW
            if (std::abs(cur_offset) < cfg.return_tol)
                transition_to(OvertakeState::FOLLOW, now);
            break;
    }

    return state_ != prev;
}
