using Plots
using LinearAlgebra
using StaticArrays
import Printf

function animate_vtol(t_hist, sol, u_hist, ref_traj; z_offset=0.0,  fps = 30, filename = "vtol_animation.gif")
    gr()

    # Geometry and scaling
    body_length = 2.0
    body_width = 0.5
    arm_length = 1.0
    thrust_scale = 5.0
    vel_scale = 0.5
    prop_offset = SVector(1.0, 0.0)

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
        tail_center_body = SVector(-body_length/2 - 0.2, 0.0) # small offset behind fuselage
        tail_len = 0.5
        el_len = 0.35
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
