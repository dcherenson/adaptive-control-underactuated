module ControlAllocation
import NaNMath
using StaticArrays, ForwardDiff, LinearAlgebra
import Main.VTOL: VTOLParams, σ_stall, aoa, Va2, fpa, R, g
import Main.HighLevelController: HighLevelParams, high_level_control, RefTrajParams, ref_pose
import Main.Adaptation: AdaptationParams, W_dot


const n_x = 6 # state dimension
const n_u = 5 # augmented control input + 2 slack variables dimension
const n_s = 0
const n_λ = 3 # lagrange multiplier dimension
const n_W = 9 # parameters dimension



@kwdef struct ControlAllocationParams
    elev_limits::Tuple{Float64, Float64} = (-deg2rad(35.0), deg2rad(35.0)) # elevator angle limits
    pitch_cmd_limits::Tuple{Float64, Float64} = (-deg2rad(20.0), deg2rad(20.0)) # pitch command limits
    log_scalar::Float64 = 0.01
    opti_weights::SVector{n_u-n_s,Float64} = @SVector[10.0, 10.0, 1.0, 0.1, 50.0]
    Γ_uλ::Float64 = 50.0

end

function τ0(x,u,pitch,p::VTOLParams)
    α_reduced = aoa(x, pitch)
    σ_reduced = σ_stall(α_reduced, p.α_stall, p.M_sigmoid)
    α = aoa(x, x[3])
    σ = σ_stall(α, p.α_stall, p.M_sigmoid)
    M = Diagonal(@SVector[p.m, p.m, p.J])

    F_aero = 0.5*p.ρ*p.S*Va2(x)*@SVector[
        -p.CD0; # drag force in wind X
        p.CL0*(1-σ_reduced) + σ_reduced*2*sign(α_reduced)*sin(α_reduced)^2*cos(α_reduced); # lift force in wind Z
        ]

    F_prop = @SVector[
        p.max_thrust_horz*u[3]^2; # body X
        p.max_thrust_vert*(u[1]^2 + u[2]^2)  # body Z
        ]

    M_aero = 0.5*p.ρ*p.S*p.c*Va2(x)*(p.Cm0*(1-σ) + σ*-2*sign(α)*sin(α)^2*cos(α))

    M_prop = p.l_motor*p.max_thrust_vert*(u[1]^2 - u[2]^2)

    return M\[R(fpa(x))*F_aero + R(pitch)*F_prop + @SVector[0.0; -p.m*g]; M_aero + M_prop]
end

function ϕ(x,u,pitch,p::VTOLParams)
    α_reduced = aoa(x, pitch)
    σ_reduced = σ_stall(α_reduced, p.α_stall, p.M_sigmoid)
    α = aoa(x, x[3])
    σ = σ_stall(α, p.α_stall, p.M_sigmoid)
    M = Diagonal(@SVector[p.m, p.m, p.J])

    F_aero = 0.5*p.ρ*p.S*Va2(x)*@SMatrix[[-u[4];; -α_reduced^2;; -(u[1]+u[2]);; 0;; 0];[0;; 0;; 0;; u[4];; α_reduced*(1-σ_reduced)]]

    M_aero = [0.5*p.ρ*p.S*p.c*Va2(x)*@SMatrix[u[4];; (1-σ)*α] 0.25*p.ρ*p.S*p.c^2*sqrt(Va2(x))*x[6]*p.CMq]

    F = [R(fpa(x))*F_aero (@SMatrix zeros(2,3)) R(pitch)*@SVector[0.0; -p.max_thrust_vert*u[2]^2*u[3]^2]]
    Moment = [(@SMatrix zeros(1,5)) M_aero @SVector[p.l_motor*p.max_thrust_vert*u[2]^2*u[3]^2]]
    return M\[F; Moment]

end

function τ_full(x,u,W,p::VTOLParams)
    return τ0(x,u,x[3],p) + ϕ(x,u,x[3],p)*W
end

function τ_reduced(x,u,W,p::VTOLParams)
    known = τ0(x,u,u[5],p) 
    est = ϕ(x,u,u[5],p)*W
    return known + est
end


function J(t,u::AbstractVector, c::ControlAllocationParams, r::RefTrajParams)
  u_diff = u[SOneTo(5)] - @SVector[0.0; 0.0; 0.0; 0.0; ref_pose(t,r)[3]] # penalize deviation from reference angle
  quad_cost =  0.5*u_diff'*(c.opti_weights.*u_diff)                     # ½‖u‖²
  barrier_cost = -c.log_scalar*(
                      NaNMath.log(u[1]*(-u[1]+1)) # front rotor throttle
                    + NaNMath.log(u[2]*(-u[2]+1)) # rear rotor throttle
                    + NaNMath.log(u[3]*(-u[3]+1)) # forward rotor throttle
                    + NaNMath.log(c.pitch_cmd_limits[1]^2-u[5]^2) # pitch command
                    + NaNMath.log(c.elev_limits[1]^2-u[4]^2) # elevator angle
                    )                     # log barrier
  return quad_cost + barrier_cost #+ 100*(u[6]^2+u[7]^2)
end

function constraint(t,x,u,W,xhat,s) 
  cons = high_level_control(t,x,u,xhat,s.high_level,s.adaptation) - τ_reduced(x,u,W,s.vtol)
  return cons
end

function L(t,x::AbstractVector,u::AbstractVector,λ::AbstractVector, W::AbstractVector, xhat::AbstractVector, s)
  dual_term = dot(λ, constraint(t,x,u,W,xhat,s))
  return J(t,u, s.control_alloc, s.high_level.ref_traj) + dual_term
end

function dL_du_jacobian(t, x::AbstractVector,u::AbstractVector,λ::AbstractVector, W::AbstractVector, xhat::AbstractVector,s)
  return ForwardDiff.jacobian(vcat(x,u,W,xhat,@SVector[t])) do v
      x_ = v[SA[1:n_x...]]
      u_ = v[SA[n_x+1 : n_x+n_u...]]
      W_ = v[SA[n_x+n_u+1 : n_x+n_u+n_W...]]
      xhat_ = v[SA[n_x+n_u+n_W+1 : n_x+n_u+n_W+n_x...]]
      t_ = v[end]
    return dL_du(t_, x_, u_, λ, W_, xhat_, s)
  end
end

function dL_dλ_jacobian(t, x::AbstractVector,u::AbstractVector,λ::AbstractVector, W::AbstractVector, xhat::AbstractVector, s)
  return ForwardDiff.jacobian(vcat(x,u,W,xhat,@SVector[t])) do v
      x_ = v[SA[1:n_x...]]
      u_ = v[SA[n_x+1 : n_x+n_u...]]
      W_ = v[SA[n_x+n_u+1 : n_x+n_u+n_W...]]
      xhat_ = v[SA[n_x+n_u+n_W+1 : n_x+n_u+n_W+n_x...]]
      t_ = v[end]
    return constraint(t_, x_, u_, W_, xhat_, s)
  end
end

function dL_du(t,x::AbstractVector,u::AbstractVector,λ::AbstractVector, W::AbstractVector, xhat::AbstractVector, s)
  ForwardDiff.gradient(u -> L(t,x,u, λ, W, xhat, s), u)
end

# function L_flat(v, s)
#   x = v[SA[1:n_x...]]
#   u = v[SA[n_x+1 : n_x+n_u...]]
#   λ = v[SA[n_x+n_u+1 : n_x+n_u+n_λ...]]
#   W = v[SA[n_x+n_u+n_λ+1 : n_x+n_u+n_λ+n_W...]]
#   xhat = v[SA[n_x+n_u+n_λ+n_W+1 : n_x+n_u+n_λ+n_W+n_x...]]
#   t = v[end]
#   return L(t, x, u, λ, W, xhat, s)
# end

function xdot(x,u,W,p::VTOLParams)
    return vcat(x[4:6], τ_full(x,u,W,p))
end


function uλW_dot(t,x,u,λ,W,xhat, xhatdot, s)

    dLduλ = [dL_du(t,x,u,λ,W,xhat,s); constraint(t,x,u,W,xhat,s)]
    d2Ldu = dL_du_jacobian(t, x, u, λ, W, xhat,s)
    d2Ldλ = dL_dλ_jacobian(t, x, u, λ, W, xhat,s)
    

    d2Ldu2 = d2Ldu[:, SA[n_x+1:n_x+n_u...]]
    d2Ldudλ = d2Ldλ[:, SA[n_x+1:n_x+n_u...]]

    H_uλ = [[d2Ldu2 d2Ldudλ'];
            [d2Ldudλ (@SMatrix zeros(n_λ, n_λ))]]

    d2Ldxdudλ = [d2Ldu[:,SA[1:n_x...]]; d2Ldλ[:,SA[1:n_x...]]]

    d2Ldxhatdudλ = [d2Ldu[:,SA[n_x+n_u+n_W+1:n_x+n_u+n_W+n_x...]]; d2Ldλ[:,SA[n_x+n_u+n_W+1:n_x+n_u+n_W+n_x...]]]

    Wdot = W_dot(x,u,W,d2Ldxdudλ[:,SA[4:6...]], dLduλ, xhat, s.adaptation, s.vtol)

    if LinearAlgebra.det(d2Ldu2) < 0.001
        uff = @MVector zeros(n_u+n_λ)
    else

        d2LdWdudλ = [d2Ldu[:,SA[n_x+n_u+1:n_x+n_u+n_W...]]; d2Ldλ[:,SA[n_x+n_u+1:n_x+n_u+n_W...]]]

        d2Ldtdudλ = [d2Ldu[:,end]; d2Ldλ[:,end]]

        uff = d2Ldtdudλ + d2Ldxdudλ * xdot(x, u, W,s.vtol) + d2Ldxhatdudλ*xhatdot + d2LdWdudλ*Wdot
    end

    uλdot = -H_uλ\(s.control_alloc.Γ_uλ*dLduλ + uff)

    return vcat(uλdot,Wdot)


end












end