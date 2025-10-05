
module Adaptation
using LinearAlgebra
import Main.VTOL: VTOLParams

@kwdef struct AdaptationParams
    K_sp::Float64 = 5.0
    K_sv::Float64 = 5.0
    Γ_e::Float64 = 100.0
    Γ_W_inv::Float64 = 0.1

end

function xhat_dot(xhat, x, u, W, p::AdaptationParams, v::VTOLParams)
    return vcat(xhat[4:6], Main.ControlAllocation.τ_full(x,u,W,v) + p.K_sp*(x[1:3] - xhat[1:3]) + p.K_sv*(x[4:6] - xhat[4:6]))
end

function W_dot(x,u,W,d2Ldxdudλ, dLduλ, xhat, p::AdaptationParams, v::VTOLParams)
    e_s = (x - xhat)[4:6]

    Wdot = p.Γ_W_inv*Main.ControlAllocation.ϕ(x,u,x[3],v)'*(p.Γ_e*e_s + d2Ldxdudλ'*dLduλ)

    max_norm = 8.0
    tol = 0.1
    fval = ((1+tol)*norm(W)^2 - max_norm^2)/(tol*max_norm^2) # constraint on θ norm
    grad_fval = 2*(1+tol)*W/(tol*max_norm^2) # gradient of constraint on θ norm
    if fval > 0.0 && Wdot'*grad_fval > 0.0
        Wdot -= p.Γ_W_inv*grad_fval*grad_fval'/(grad_fval'*p.Γ_W_inv*grad_fval)*Wdot*fval
    end
    return Wdot

end


end