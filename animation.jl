using Plots
using LinearAlgebra
using StaticArrays
import Printf

function animate_vtol(t_hist, sol, u_hist, ref_traj; z_offset=0.0,  fps = 30, filename = "vtol_animation.gif", vehicle_scale=1.6)
    gr()

    # Geometry and scaling
    body_length = 2.0 * vehicle_scale
    body_width = 0.5 * vehicle_scale
    arm_length = 1.0 * vehicle_scale
    thrust_scale = 5.0
    vel_scale = 0.5
    prop_offset = SVector(1.0 * vehicle_scale, 0.0)

    dt = t_hist[2] - t_hist[1]
    skip = round(Int,1.0 / (fps * dt))

    ylims = (minimum(sol(t)[2] + z_offset for t in t_hist) - 1, maximum(sol(t)[2] + z_offset for t in t_hist) + 1)

    trajx = [ref_traj(t)[1] for t in t_hist]
    trajy = [ref_traj(t)[2] + z_offset for t in t_hist]

    xhist = []
    yhist = []
    anim = @animate for i in 1:skip:length(t_hist)
        t = t_hist[i]
        state = sol(t)
        v = state[1]     # horizontal position
        z = state[2] + z_offset     # vertical position (altitude)
        xhist = push!(xhist, v)
        yhist = push!(yhist, z)
        vx = state[4]
        vz = state[5]
        θ = state[3]

        T_right, T_left, T_prop, elev_def = u_hist[1:4, i]

        # Rotation matrix: body to world
        R = SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))

        # Convert body-frame velocity to world-frame
        vel_world = SVector(vx, vz)

        # Body shape
        body = [
            SVector(-body_length/2, -body_width/2),
            SVector( body_length/2, -body_width/2),
            SVector( body_length/2,  body_width/2),
            SVector(-body_length/2,  body_width/2),
            SVector(-body_length/2, -body_width/2)
        ]
        body_world = [R * p .+ SVector(v, z) for p in body]
        bx = [p[1] for p in body_world]
        by = [p[2] for p in body_world]

        # Rotor and propeller positions
        left_pos  = R * SVector(-arm_length, 0.0) .+ SVector(v, z)
        right_pos = R * SVector( arm_length, 0.0) .+ SVector(v, z)
        prop_pos  = R * prop_offset .+ SVector(v, z)

        # Thrust directions in world frame
        thrust_y = R * SVector(0.0, 1.0)
        thrust_x = R * SVector(1.0, 0.0)

        # Arrows
        left_arrow  = thrust_scale * T_left * thrust_y
        right_arrow = thrust_scale * T_right * thrust_y
        prop_arrow  = thrust_scale * T_prop * thrust_x
        vel_arrow   = vel_scale * vel_world

        # Plot
        plot(bx, by, lw=2, label="", aspect_ratio=1, xlims=(v-20.0,v+20.0), ylims=(z-5.0,z+5.0), size=(1500,500))
        scatter!([left_pos[1], right_pos[1], prop_pos[1]], [left_pos[2], right_pos[2], prop_pos[2]], label="", color=:black)

        # Thrust arrows
        quiver!([left_pos[1]], [left_pos[2]], quiver=([left_arrow[1]], [left_arrow[2]]), color=:red, label="")
        quiver!([right_pos[1]], [right_pos[2]], quiver=([right_arrow[1]], [right_arrow[2]]), color=:red, label="")
        quiver!([prop_pos[1]], [prop_pos[2]], quiver=([prop_arrow[1]], [prop_arrow[2]]), color=:orange, label="")

        # Velocity vector (center of mass)
        quiver!([v], [z], quiver=([vel_arrow[1]], [vel_arrow[2]]), color=:blue)

        plot!(trajx, trajy, label="Reference Trajectory", color=:green, lw=1.5, linestyle=:dash)

        plot!([ref_traj(t)[1]], [ref_traj(t)[2]], seriestype=:scatter, markersize=5, color=:green, label="Reference Position")

        plot!(xhist, yhist, label="VTOL Trace", color=:purple, lw=2, linestyle=:dash)

        plot!([-10000],[-10000], color = :red, label="Vertical Thrust")
        plot!([-10000],[-10000], color = :orange, label="Pusher Thrust")
        plot!([-10000],[-10000], color = :blue, label="Velocity Vector")

        # Tailplane (horizontal stabilizer) and elevator
        tail_center_body = SVector(-body_length/2 - 0.2 * vehicle_scale, 0.0) # small offset behind fuselage
        tail_len = 0.5 * vehicle_scale
        el_len = 0.35 * vehicle_scale
        # hinge at trailing edge of tail (in body frame)
        hinge_body = tail_center_body - SVector(tail_len/2, 0.0)
        # tail endpoints in body frame
        tail_a = tail_center_body + SVector(-tail_len/2, 0.0)
        tail_b = tail_center_body + SVector(tail_len/2, 0.0)

        # elevator deflection (assume elev is in radians, positive down)
        Rot_elev = SMatrix{2,2,Float64}(cos(elev_def), sin(elev_def), -sin(elev_def), cos(elev_def))

        # elevator end point in body frame (relative to hinge)
        el_tip_rel = Rot_elev * SVector(-el_len, 0.0)

        # world transforms
        hinge_world = R * hinge_body .+ SVector(v, z)
        tail_a_world = R * tail_a .+ SVector(v, z)
        tail_b_world = R * tail_b .+ SVector(v, z)
        el_tip_world = R * (hinge_body + el_tip_rel) .+ SVector(v, z)

        # draw tailplane and elevator
        plot!([tail_a_world[1], tail_b_world[1]], [tail_a_world[2], tail_b_world[2]], lw=3, color=:black, label=false)
        plot!([hinge_world[1], el_tip_world[1]], [hinge_world[2], el_tip_world[2]], lw=4, color=:darkgray, label=false)

        plot!(legendfontsize=14,  xtickfontsize=14, ytickfontsize=14, legend =:topright)

        title!("t = $(Printf.@sprintf("%2.2f", t)) s, V = $(Printf.@sprintf("%2.2f", norm(vel_world))) m/s")
    end

    gif(anim, filename, fps=fps)
end

function _case_u_hist(case)
    if hasproperty(case, :u_hist)
        return getproperty(case, :u_hist)
    elseif hasproperty(case, :data) && hasproperty(getproperty(case, :data), :u_hist)
        return getproperty(getproperty(case, :data), :u_hist)
    end

    throw(ArgumentError("case must expose either u_hist or data.u_hist"))
end

function _control_at(u_hist, t_hist, t)
    i = clamp(searchsortedlast(t_hist, t), 1, length(t_hist))
    if u_hist isa AbstractMatrix
        return @SVector[u_hist[1, i], u_hist[2, i], u_hist[3, i], u_hist[4, i], u_hist[5, i]]
    end

    return u_hist[i]
end

function _draw_vtol!(state, u; z_offset=0.0, color=:blue, label="", thrust_scale=5.0, vel_scale=0.5, vehicle_scale=1.6)
    body_length = 2.0 * vehicle_scale
    body_width = 0.5 * vehicle_scale
    arm_length = 1.0 * vehicle_scale
    prop_offset = SVector(1.0 * vehicle_scale, 0.0)

    v = state[1]
    z = state[2] + z_offset
    vx = state[4]
    vz = state[5]
    theta = state[3]

    T_right, T_left, T_prop, elev_def = u[1:4]

    R = SMatrix{2,2}(cos(theta), sin(theta), -sin(theta), cos(theta))
    center = SVector(v, z)

    body = [
        SVector(-body_length/2, -body_width/2),
        SVector( body_length/2, -body_width/2),
        SVector( body_length/2,  body_width/2),
        SVector(-body_length/2,  body_width/2),
        SVector(-body_length/2, -body_width/2),
    ]
    body_world = [R * p .+ center for p in body]
    bx = [p[1] for p in body_world]
    by = [p[2] for p in body_world]

    left_pos = R * SVector(-arm_length, 0.0) .+ center
    right_pos = R * SVector(arm_length, 0.0) .+ center
    prop_pos = R * prop_offset .+ center

    thrust_y = R * SVector(0.0, 1.0)
    thrust_x = R * SVector(1.0, 0.0)
    left_arrow = thrust_scale * T_left * thrust_y
    right_arrow = thrust_scale * T_right * thrust_y
    prop_arrow = thrust_scale * T_prop * thrust_x
    vel_arrow = vel_scale * SVector(vx, vz)

    plot!(bx, by, lw=3, color=color, label=label)
    scatter!(
        [left_pos[1], right_pos[1], prop_pos[1]],
        [left_pos[2], right_pos[2], prop_pos[2]],
        label="", color=color, markersize=4,
    )
    quiver!([left_pos[1]], [left_pos[2]], quiver=([left_arrow[1]], [left_arrow[2]]), color=:red, label="")
    quiver!([right_pos[1]], [right_pos[2]], quiver=([right_arrow[1]], [right_arrow[2]]), color=:red, label="")
    quiver!([prop_pos[1]], [prop_pos[2]], quiver=([prop_arrow[1]], [prop_arrow[2]]), color=:orange, label="")
    quiver!([v], [z], quiver=([vel_arrow[1]], [vel_arrow[2]]), color=:blue, label="")

    tail_center_body = SVector(-body_length/2 - 0.2 * vehicle_scale, 0.0)
    tail_len = 0.5 * vehicle_scale
    el_len = 0.35 * vehicle_scale
    hinge_body = tail_center_body - SVector(tail_len/2, 0.0)
    tail_a = tail_center_body + SVector(-tail_len/2, 0.0)
    tail_b = tail_center_body + SVector(tail_len/2, 0.0)
    Rot_elev = SMatrix{2,2,Float64}(cos(elev_def), sin(elev_def), -sin(elev_def), cos(elev_def))
    el_tip_rel = Rot_elev * SVector(-el_len, 0.0)

    hinge_world = R * hinge_body .+ center
    tail_a_world = R * tail_a .+ center
    tail_b_world = R * tail_b .+ center
    el_tip_world = R * (hinge_body + el_tip_rel) .+ center

    plot!([tail_a_world[1], tail_b_world[1]], [tail_a_world[2], tail_b_world[2]], lw=3, color=color, label="")
    plot!([hinge_world[1], el_tip_world[1]], [hinge_world[2], el_tip_world[2]], lw=4, color=color, label="")
end

function animate_vtol_comparison(
    adaptive_case,
    pid_case,
    ref_traj;
    z_offset=0.0,
    fps=15,
    filename="vtol_landing_adaptive_vs_pid.gif",
    labels=("Adaptive", "PID"),
    colors=(:blue, :red),
    x_window=90.0,
    z_window=25.0,
    vehicle_scale=1.6,
    size=(1400, 400),
)
    gr()

    adaptive_t = getproperty(adaptive_case, :t_vec)
    pid_t = getproperty(pid_case, :t_vec)
    adaptive_sol = getproperty(adaptive_case, :sol)
    pid_sol = getproperty(pid_case, :sol)
    adaptive_u_hist = _case_u_hist(adaptive_case)
    pid_u_hist = _case_u_hist(pid_case)

    t_start = max(first(adaptive_t), first(pid_t))
    t_stop = min(last(adaptive_t), last(pid_t))
    frame_times = collect(t_start:(1 / fps):t_stop)
    ref_times = range(t_start, t_stop; length=400)
    ref_x = [ref_traj(t)[1] for t in ref_times]
    ref_z = [ref_traj(t)[2] + z_offset for t in ref_times]

    adaptive_xhist = Float64[]
    adaptive_zhist = Float64[]
    pid_xhist = Float64[]
    pid_zhist = Float64[]

    anim = @animate for t in frame_times
        adaptive_state = adaptive_sol(t)
        pid_state = pid_sol(t)
        adaptive_u = _control_at(adaptive_u_hist, adaptive_t, t)
        pid_u = _control_at(pid_u_hist, pid_t, t)
        ref_state = ref_traj(t)

        push!(adaptive_xhist, adaptive_state[1])
        push!(adaptive_zhist, adaptive_state[2] + z_offset)
        push!(pid_xhist, pid_state[1])
        push!(pid_zhist, pid_state[2] + z_offset)

        x_left = minimum((adaptive_state[1], pid_state[1], ref_state[1]))
        x_right = maximum((adaptive_state[1], pid_state[1], ref_state[1]))
        x_center = 0.5 * (x_left + x_right)
        x_min = x_center - x_window/2
        x_max = x_center + x_window/2

        z_low = minimum((adaptive_state[2] + z_offset, pid_state[2] + z_offset, ref_state[2] + z_offset))
        z_high = maximum((adaptive_state[2] + z_offset, pid_state[2] + z_offset, ref_state[2] + z_offset))
        z_center = 0.5 * (z_low + z_high)
        z_min = z_center - z_window/2
        z_max = z_center + z_window/2

        plot(
            ref_x, ref_z,
            label="", color=:black, lw=2, linestyle=:dash,
            xlims=(x_min, x_max), ylims=(z_min, z_max),
            aspect_ratio=:equal, size=size, grid=true,
            xlabel="p_x [m]", ylabel="p_z [m]",
            legend=:topright, legendfontsize=11,
            xtickfontsize=11, ytickfontsize=11,
        )
        scatter!([ref_state[1]], [ref_state[2] + z_offset], color=:black, markersize=4, label="")

        plot!(adaptive_xhist, adaptive_zhist, label="", color=colors[1], lw=2)
        plot!(pid_xhist, pid_zhist, label="", color=colors[2], lw=2, linestyle=:dash)

        _draw_vtol!(adaptive_state, adaptive_u; z_offset=z_offset, color=colors[1], label="", vehicle_scale=vehicle_scale)
        _draw_vtol!(pid_state, pid_u; z_offset=z_offset, color=colors[2], label="", vehicle_scale=vehicle_scale)

        plot!([-10000], [-10000], color=:black, lw=2, linestyle=:dash, label="Reference Trajectory")
        scatter!([-10000], [-10000], color=:black, markersize=4, label="Reference Position")
        plot!([-10000], [-10000], color=colors[1], lw=2, label="$(labels[1]) Trace")
        plot!([-10000], [-10000], color=colors[2], lw=2, linestyle=:dash, label="$(labels[2]) Trace")
        plot!([-10000], [-10000], color=:red, label="Vertical Thrust")
        plot!([-10000], [-10000], color=:orange, label="Pusher Thrust")
        plot!([-10000], [-10000], color=:blue, label="Velocity Vector")

        adaptive_speed = norm(SVector(adaptive_state[4], adaptive_state[5]))
        pid_speed = norm(SVector(pid_state[4], pid_state[5]))
        title!(
            "Landing comparison | t = $(Printf.@sprintf("%2.2f", t)) s | " *
            "$(labels[1]) V = $(Printf.@sprintf("%2.2f", adaptive_speed)) m/s | " *
            "$(labels[2]) V = $(Printf.@sprintf("%2.2f", pid_speed)) m/s"
        )
    end

    gif(anim, filename, fps=fps)
end
