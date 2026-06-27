#include "../ov_inc/offset_planner.hpp"
#include <algorithm>
#include <cmath>

void OffsetPlanner::step(OvertakeState state, const Config & cfg, double dt)
{
    goal_ = (state == OvertakeState::OVERTAKE) ? cfg.overtake_offset : 0.0;

    double max_delta = cfg.offset_rate_limit * std::max(dt, 0.001);
    double diff      = goal_ - offset_;

    if (std::abs(diff) <= max_delta)
        offset_ = goal_;
    else
        offset_ += (diff > 0) ? max_delta : -max_delta;
}
