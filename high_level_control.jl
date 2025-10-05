
module HighLevelController
using StaticArrays, ForwardDiff, LinearAlgebra
import Main.Adaptation: AdaptationParams



@kwdef struct RefTrajParams
    t_start_land::Float64 = 3.0
    v_ref_0::Float64 = 20.0
    h_ref_0::Float64 = 100.0
    a_ref::Float64 = -1.0
    descent_rate::Float64 = -0.75
    W_ref_0::Float64 = 0.0
    W_ref_final::Float64 = deg2rad(15.0)
end


@kwdef struct HighLevelParams
    kp::Float64 = 0.5
    kd::Float64 = 2*sqrt(2)*kp
    ϵ::Float64 = 0.2
    ref_traj::RefTrajParams = RefTrajParams()
end

σ(t,t_0) = 0.5*(1.0 + tanh((t-t_0)))

function ref_pose(t, params::RefTrajParams)
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
       (1-s2)*deg2rad(params.W_ref_0) + s2*deg2rad(params.W_ref_final)
    ]
end

ref_velocity(t, params::RefTrajParams) = ForwardDiff.derivative(t -> ref_pose(t, params), t)
ref_accel(t, params::RefTrajParams) = ForwardDiff.derivative(t -> ref_velocity(t, params), t)

function high_level_control(t,x,u,xhat, h::HighLevelParams, a::AdaptationParams)

    Kp = Diagonal(@SVector[h.kp, h.kp, h.kp/h.ϵ^2])
    Kd = Diagonal(@SVector[h.kd, h.kd, h.kd/h.ϵ])

    p_err = xhat[SOneTo(3)] - [ref_pose(t,h.ref_traj)[SOneTo(2)]; u[5] + xhat[3] - x[3]]
    d_err = xhat[4:6] - [ref_velocity(t,h.ref_traj)[SOneTo(2)];0.0]
    command = -Kp*p_err - Kd*d_err + [ref_accel(t,h.ref_traj)[SOneTo(2)];0.0]
    command += -a.K_sp*(x[SOneTo(3)] - xhat[SOneTo(3)]) - a.K_sv*(x[4:6] - xhat[4:6])
    return clamp.(command, @SVector[-2,-100,-500], @SVector[5,100,500]) + @SVector[u[6];u[7];0]
end

end