module Simulation
using StaticArrays
import Main.VTOL: VTOLParams, dynamics_sim
import Main.ControlAllocation: ControlAllocationParams, uλW_dot, n_u, n_λ, n_W, n_x
import Main.HighLevelController: HighLevelParams
import Main.Adaptation: AdaptationParams, xhat_dot

@kwdef struct SimulationParams
    vtol::VTOLParams = VTOLParams()
    control_alloc::ControlAllocationParams = ControlAllocationParams()
    high_level::HighLevelParams = HighLevelParams()
    adaptation::AdaptationParams = AdaptationParams()
    dt::Float64 = 0.01
    t0::Float64 = 0.0
    t_final::Float64 = 30.0
end

@kwdef struct SimulationData{F}
    u::MVector{n_u,F}
    λ::MVector{n_λ,F}
    W::MVector{n_W,F}
    xhat::MVector{n_x,F}
    params::SimulationParams = SimulationParams()
    u_hist::Vector{MVector{n_u,F}} = [u]
    λ_hist::Vector{MVector{n_λ,F}} = [λ]
    W_hist::Vector{MVector{n_W,F}} = [W]
    xhat_hist::Vector{MVector{n_x,F}} = [xhat]
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
    
    push!(integrator.p.u_hist, u)
    push!(integrator.p.λ_hist, λ)
    push!(integrator.p.W_hist, W)
    push!(integrator.p.xhat_hist, xhat)
    push!(integrator.p.t_hist, t+dt)

end


end