#pragma once
#include "state_machine.hpp"

class OffsetPlanner
{
public:
    struct Config {
        double overtake_offset{-0.534};
        double offset_rate_limit{0.25};
    };

    // Tính toán target_offset rate-limited theo state hiện tại.
    // Gọi mỗi chu kỳ với dt [s].
    void step(OvertakeState state, const Config & cfg, double dt);

    double offset() const { return offset_; }

private:
    double offset_{0.0};
    double goal_{0.0};
};
