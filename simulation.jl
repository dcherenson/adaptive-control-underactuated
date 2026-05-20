module Simulation
using StaticArrays
import Main.VTOL: VTOLParams, angle_normalize, dynamics_sim, g, R, Va2
import Main.ControlAllocation: ControlAllocationParams, uλW_dot, n_u, n_λ, n_W, n_x
import Main.HighLevelController: HighLevelParams, LandingTrajParams, TakeoffTrajParams, SinusoidalTransitionTrajParams, HoverTrajParams, high_level_control, ref_pose
import Main.Adaptation: AdaptationParams, xhat_dot

@kwdef struct PIDGains
    kp::Float64
    ki::Float64
    kd::Float64
    i_limit::Float64
    out_limit::Float64
end

@kwdef struct CascadedPIDParams
    # Outer loop: pitch angle error -> pitch rate command
    theta_hover::PIDGains = PIDGains(kp=6.228845113923736, ki=0.7627180213961051, kd=1.610646975999857, i_limit=0.21235433329717518, out_limit=2.607409204103294)
    theta_fw::PIDGains = PIDGains(kp=4.140902579583811, ki=0.21548281390889493, kd=0.19465227221502693, i_limit=0.12829925812585746, out_limit=0.996981866672961)
    # Inner loop: pitch rate error -> pitch moment command
    q_hover::PIDGains = PIDGains(kp=12.540123852871549, ki=0.6534211743132547, kd=0.12632870115792455, i_limit=0.30452218869016134, out_limit=46.14207499618037)
    q_fw::PIDGains = PIDGains(kp=2.283473956525624, ki=0.7075176757989048, kd=0.04184332007633861, i_limit=0.40835642307919157, out_limit=60.0)
    hover_speed::Float64 = 8.519220520703417
    fixed_wing_speed::Float64 = 10.519220520703417
    moment_to_elevator_gain::Float64 = -0.014645461382071314
    fw_ref_pitch_weight::Float64 = 0.40
    fw_accel_to_pitch_gain::Float64 = -0.12
    ax_to_elevator_gain::Float64 = 0.0
    u_rate_limits::SVector{n_u,Float64} = @SVector[2.0, 2.0, 2.0, 4.0, 4.0]
    throttle_sum_limit::Float64 = 1.95
end

@kwdef struct SimulationParams
    vtol::VTOLParams = VTOLParams()
    control_alloc::ControlAllocationParams = ControlAllocationParams()
    high_level::HighLevelParams = HighLevelParams()
    adaptation::AdaptationParams = AdaptationParams()
    pid::CascadedPIDParams = CascadedPIDParams()
    dt::Float64 = 0.01
    t0::Float64 = 0.0
    t_final::Float64 = 30.0
end

function scenario_ref_traj(scenario::Symbol)
    if scenario == :landing
        return LandingTrajParams()
    elseif scenario == :takeoff
        return TakeoffTrajParams()
    elseif scenario == :sinusoidal_transition
        return SinusoidalTransitionTrajParams()
    elseif scenario == :hover
        return HoverTrajParams()
    else
        throw(ArgumentError("Unknown scenario '$scenario'. Use :landing, :takeoff, or :sinusoidal_transition."))
    end
end

function scenario_high_level_params(scenario::Symbol)
    return HighLevelParams(ref_traj=scenario_ref_traj(scenario))
end

@kwdef mutable struct SimulationData{F}
    u::MVector{n_u,F}
    λ::MVector{n_λ,F}
    W::MVector{n_W,F}
    xhat::MVector{n_x,F}
    theta_int::F = 0.0
    q_int::F = 0.0
    prev_q_err::F = 0.0
    params::SimulationParams = SimulationParams()
    u_hist::Vector{MVector{n_u,F}} = [copy(u)]
    λ_hist::Vector{MVector{n_λ,F}} = [copy(λ)]
    W_hist::Vector{MVector{n_W,F}} = [copy(W)]
    xhat_hist::Vector{MVector{n_x,F}} = [copy(xhat)]
    t_hist::Vector{F} = [params.t0]
end


function dudt!(dv,v,p,t)
    dv[1:n_x] = dynamics_sim(v, p.u, p.params.vtol)
end

function adaptive_control_allocator!(integrator)
    x      = integrator.u
    u      = integrator.p.u
    λ      = integrator.p.λ
    W      = integrator.p.W
    xhat   = integrator.p.xhat
    t      = integrator.t
    dt     = integrator.p.params.dt
    xhatdot = xhat_dot(xhat, x, u, W, integrator.p.params.adaptation, integrator.p.params.vtol)
    uλWdot = uλW_dot(t,x,u, λ, W,xhat,xhatdot,integrator.p.params)

    udot = uλWdot[1:n_u]
    udot = clamp.(udot, @MVector[-5.0, -5.0, -2.0, -2π, -deg2rad(60.0)], @MVector[5.0, 5.0, 1.0, 2π, deg2rad(60.0)])
    u += udot*dt
    λ += uλWdot[n_u+1:n_u+n_λ]*dt
    W += uλWdot[n_u+n_λ+1:n_u+n_λ+n_W]*dt
    xhat += xhatdot*dt

    u = clamp.(u, 0.001.+@MVector[0.0, 0.0, 0.0, integrator.p.params.control_alloc.elev_limits[1], integrator.p.params.control_alloc.pitch_cmd_limits[1]],
         -0.001.+@MVector[1.0, 1.0, 1.0, integrator.p.params.control_alloc.elev_limits[2], integrator.p.params.control_alloc.pitch_cmd_limits[2]])
    
    integrator.p.u[:] = u
    integrator.p.λ[:] = λ
    integrator.p.W[:] = W
    integrator.p.xhat[:] = xhat    
    
    push!(integrator.p.u_hist, copy(integrator.p.u))
    push!(integrator.p.λ_hist, copy(integrator.p.λ))
    push!(integrator.p.W_hist, copy(integrator.p.W))
    push!(integrator.p.xhat_hist, copy(integrator.p.xhat))
    push!(integrator.p.t_hist, t+dt)

end

sat(x, xlim) = clamp(x, -xlim, xlim)

smoothstep01(x) = x <= 0 ? 0.0 : (x >= 1 ? 1.0 : x*x*(3 - 2*x))

function fw_blend_factor(x, pid::CascadedPIDParams)
    v = sqrt(Va2(x))
    raw = (v - pid.hover_speed) / (pid.fixed_wing_speed - pid.hover_speed)
    return smoothstep01(raw)
end

function pid_eval(err, derr, integ, gains::PIDGains, dt)
    integ_new = clamp(integ + err*dt, -gains.i_limit, gains.i_limit)
    out = gains.kp*err + gains.ki*integ_new + gains.kd*derr
    return sat(out, gains.out_limit), integ_new
end

function pid_inner_loop_target(t, x, u, cmd, params::SimulationParams, data)
    vtol = params.vtol
    pid = params.pid
    ca = params.control_alloc

    # Blended reference pitch: tilt-to-accelerate for hover, trajectory pitch for fixed-wing.
    β = fw_blend_factor(x, pid)
    # Positive pitch rotates vertical thrust toward negative inertial x,
    # so hover lateral acceleration requires opposite-sign pitch command.
    θ_hover = atan(-cmd[1], cmd[2] + g)
    θ_ref = ref_pose(t, params.high_level.ref_traj)[3]
    θ_fw = pid.fw_ref_pitch_weight*θ_ref + (1 - pid.fw_ref_pitch_weight)*θ_hover + pid.fw_accel_to_pitch_gain*cmd[1]
    θ_cmd = (1 - β)*θ_hover + β*θ_fw

    eθ = angle_normalize(θ_cmd - x[3])
    q = x[6]

    q_cmd_hover, theta_int_h = pid_eval(eθ, -q, data.theta_int, pid.theta_hover, params.dt)
    q_cmd_fw, theta_int_f = pid_eval(eθ, -q, data.theta_int, pid.theta_fw, params.dt)
    q_cmd = (1 - β)*q_cmd_hover + β*q_cmd_fw
    data.theta_int = (1 - β)*theta_int_h + β*theta_int_f

    q_err = q_cmd - q
    q_err_dot = (q_err - data.prev_q_err) / params.dt
    data.prev_q_err = q_err

    m_hover, q_int_h = pid_eval(q_err, q_err_dot, data.q_int, pid.q_hover, params.dt)
    m_fw, q_int_f = pid_eval(q_err, q_err_dot, data.q_int, pid.q_fw, params.dt)
    m_cmd = (1 - β)*m_hover + β*m_fw
    data.q_int = (1 - β)*q_int_h + β*q_int_f

    # Translational command to nominal thrust request.
    F_des_inertial = vtol.m * @SVector[cmd[1], cmd[2] + g]
    F_des_body = R(-x[3]) * F_des_inertial

    u3_tgt = sqrt(clamp(max(0.0, β*F_des_body[1]) / vtol.max_thrust_horz, 0.0, 1.0))
    total_vert_sq = clamp(max(0.0, F_des_body[2]) / vtol.max_thrust_vert, 0.0, pid.throttle_sum_limit)

    diff_sq_hover = clamp(m_cmd / (vtol.l_motor*vtol.max_thrust_vert), -0.9*total_vert_sq, 0.9*total_vert_sq)
    u1_sq = 0.5*(total_vert_sq + (1 - β)*diff_sq_hover)
    u2_sq = 0.5*(total_vert_sq - (1 - β)*diff_sq_hover)

    u1_tgt = sqrt(clamp(u1_sq, 0.0, 1.0))
    u2_tgt = sqrt(clamp(u2_sq, 0.0, 1.0))

    elev_tgt = clamp(β*(pid.moment_to_elevator_gain*m_cmd + pid.ax_to_elevator_gain*cmd[1]), ca.elev_limits[1], ca.elev_limits[2])
    pitch_cmd_tgt = clamp(θ_cmd, ca.pitch_cmd_limits[1], ca.pitch_cmd_limits[2])

    return @MVector[u1_tgt, u2_tgt, u3_tgt, elev_tgt, pitch_cmd_tgt]
end

function cascaded_pid_allocator!(integrator)
    x = integrator.u
    p = integrator.p
    dt = p.params.dt
    t = integrator.t

    # Benchmark path: no adaptation, perfect state feedback for the high-level loop.
    p.xhat[:] = x
    cmd = high_level_control(t, x, p.u, p.xhat, p.params.high_level, p.params.adaptation)
    u_tgt = pid_inner_loop_target(t, x, p.u, cmd, p.params, p)

    du_max = p.params.pid.u_rate_limits * dt
    p.u += clamp.(u_tgt - p.u, -du_max, du_max)

    p.u[:] = clamp.(
        p.u,
        0.001 .+ @MVector[0.0, 0.0, 0.0, p.params.control_alloc.elev_limits[1], p.params.control_alloc.pitch_cmd_limits[1]],
        -0.001 .+ @MVector[1.0, 1.0, 1.0, p.params.control_alloc.elev_limits[2], p.params.control_alloc.pitch_cmd_limits[2]],
    )

    push!(p.u_hist, copy(p.u))
    push!(p.λ_hist, copy(p.λ))
    push!(p.W_hist, copy(p.W))
    push!(p.xhat_hist, copy(p.xhat))
    push!(p.t_hist, t + dt)
end


end
