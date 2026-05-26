
module HighLevelController
using StaticArrays, ForwardDiff, LinearAlgebra
import Main.Adaptation: AdaptationParams


abstract type AbstractRefTrajParams end

@kwdef struct HoverTrajParams <: AbstractRefTrajParams
    h_ref_0::Float64 = 100.0
    θ_ref_0::Float64 = deg2rad(3.0)
end

@kwdef struct LandingTrajParams <: AbstractRefTrajParams
    t_start_land::Float64 = 3.0
    v_ref_0::Float64 = 20.0
    h_ref_0::Float64 = 100.0
    a_ref::Float64 = -1.0
    descent_rate::Float64 = -0.75
    θ_ref_0::Float64 = 0.0
    θ_ref_final::Float64 = deg2rad(15.0)
end

@kwdef struct TakeoffTrajParams <: AbstractRefTrajParams
    t_start_takeoff::Float64 = 1.0
    v_ref_0::Float64 = 0.0
    v_ref_final::Float64 = 20.0
    h_ref_0::Float64 = 100.0
    a_ref::Float64 = 0.75
    climb_rate::Float64 = 0.0
    θ_ref_0::Float64 = deg2rad(1.0)
    θ_ref_final::Float64 = deg2rad(4.0)
end

@kwdef struct SinusoidalTransitionTrajParams <: AbstractRefTrajParams
    t_start::Float64 = 3.0
    v_ref_0::Float64 = 12.0
    h_ref_0::Float64 = 40.0
    a_ref::Float64 = 0.4
    transition_duration::Float64 = 10.0
    h_rate::Float64 = 0.0
    θ_ref_0::Float64 = deg2rad(5.0)
    θ_ref_final::Float64 = deg2rad(15.0)
    sin_amplitude::Float64 = 4.0
    sin_frequency_hz::Float64 = 0.25
    sin_phase::Float64 = 0.0
    sin_window_start::Float64 = 0.25
    sin_window_end::Float64 = 0.75
end

# Keep old name for existing scripts and notebooks.
const RefTrajParams = LandingTrajParams


@kwdef struct HighLevelParams
    kp::Float64 = 0.5
    kd::Float64 = 2*sqrt(2*kp)
    ϵ::Float64 = 0.2
    ref_traj::AbstractRefTrajParams = LandingTrajParams()
end

σ(t,t_0) = 0.5*(1.0 + tanh((t-t_0)))

function ref_pose(t, params::HoverTrajParams)
    return @SVector[
        0.0;
        params.h_ref_0;
        params.θ_ref_0
    ]
end

function ref_pose(t, params::LandingTrajParams)
    T_transition = -params.v_ref_0 / params.a_ref    
    p0 = -params.v_ref_0*params.t_start_land - 0.5*params.a_ref*T_transition^2 - params.v_ref_0*T_transition

    s1 = σ(t, params.t_start_land)
    s2 = σ(t, params.t_start_land + T_transition)

    r1 = p0 + params.v_ref_0*t
    r2 = p0 + params.v_ref_0*t + 0.5*params.a_ref*(t-params.t_start_land)^2
    r3 = 0.0

    return @SVector[
       (1-s1)*r1 + s1*((1-s2)*r2 + s2*r3);
       (1-s1)*params.h_ref_0 + s1*(params.h_ref_0+params.descent_rate*(t-params.t_start_land));
       (1-s2)*params.θ_ref_0 + s2*params.θ_ref_final
    ]
end

function ref_pose(t, params::TakeoffTrajParams)
    T_transition = (params.v_ref_final - params.v_ref_0) / params.a_ref

    s1 = σ(t, params.t_start_takeoff)
    s2 = σ(t, params.t_start_takeoff + T_transition)

    t_rel = t - params.t_start_takeoff
    r1 = 0.0
    r2 = params.v_ref_0*t_rel + 0.5*params.a_ref*t_rel^2
    r2_end = params.v_ref_0*T_transition + 0.5*params.a_ref*T_transition^2
    r3 = r2_end + params.v_ref_final*(t_rel - T_transition)

    return @SVector[
       (1-s1)*r1 + s1*((1-s2)*r2 + s2*r3);
       (1-s1)*params.h_ref_0 + s1*(params.h_ref_0+params.climb_rate*t_rel);
       (1-s2)*params.θ_ref_0 + s2*params.θ_ref_final
    ]
end

function ref_pose(t, params::SinusoidalTransitionTrajParams)
    T_transition = params.transition_duration
    t_rel = t - params.t_start

    s1 = σ(t, params.t_start)
    s2 = σ(t, params.t_start + T_transition)

    r1 = params.v_ref_0*t
    r2 = params.v_ref_0*t + 0.5*params.a_ref*t_rel^2
    r3 = params.v_ref_0*t + params.a_ref*T_transition*t_rel - 0.5*params.a_ref*T_transition^2

    z_base = (1-s1)*params.h_ref_0 + s1*(params.h_ref_0 + params.h_rate*t_rel)

    t_sin_on = params.t_start + params.sin_window_start*T_transition
    t_sin_off = params.t_start + params.sin_window_end*T_transition
    sin_window = σ(t, t_sin_on) - σ(t, t_sin_off)
    ω = 2*pi*params.sin_frequency_hz
    z_sinusoid = params.sin_amplitude * sin(ω*(t - t_sin_on) + params.sin_phase)

    return @SVector[
       (1-s1)*r1 + s1*((1-s2)*r2 + s2*r3);
       z_base + sin_window*z_sinusoid;
       (1-s2)*params.θ_ref_0 + s2*params.θ_ref_final
    ]
end

ref_velocity(t, params::AbstractRefTrajParams) = ForwardDiff.derivative(t -> ref_pose(t, params), t)
ref_accel(t, params::AbstractRefTrajParams) = ForwardDiff.derivative(t -> ref_velocity(t, params), t)

function high_level_control(t,x,u,xhat, h::HighLevelParams, a::AdaptationParams)

    Kp = Diagonal(@SVector[h.kp, h.kp, h.kp/h.ϵ^2])
    Kd = Diagonal(@SVector[h.kd, h.kd, h.kd/h.ϵ])

    p_err = xhat[SOneTo(3)] - [ref_pose(t,h.ref_traj)[SOneTo(2)]; u[5] + xhat[3] - x[3]]
    d_err = xhat[4:6] - [ref_velocity(t,h.ref_traj)[SOneTo(2)];0.0]
    command = -Kp*p_err - Kd*d_err + [ref_accel(t,h.ref_traj)[SOneTo(2)];0.0]
    command += -a.K_sp*(x[SOneTo(3)] - xhat[SOneTo(3)]) - a.K_sv*(x[4:6] - xhat[4:6])
    cmd = clamp.(command, @SVector[-2,-100,-500], @SVector[5,100,500])
    return cmd
end

end