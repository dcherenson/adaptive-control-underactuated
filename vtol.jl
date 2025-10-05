module VTOL
using StaticArrays
const g = 9.81 # gravity [m/s^2]

@kwdef struct VTOLParams
    m::Float64 = 1.0 # mass [kg]
    J::Float64 = 0.1 # moment of inertia [kg*m^2]
    l_motor::Float64 = 0.5 # distance between center and thrusters [m]
    max_thrust_vert::Float64 = 100.0 # max thrust [N]
    max_thrust_horz::Float64 = 50.0 # max torque [N*m]
    ρ::Float64 = 1.225
    S::Float64 = 0.5
    c::Float64 = 0.2
    Cmδ::Float64 = -0.99
    Cmα::Float64 = -2.74
    Cm0::Float64 = 0.0135
    CD0::Float64 = 0.043
    CDα::Float64 = 0.03
    CDδ::Float64 = 0.0135
    CDω::Float64 = 0.1
    CL0::Float64 = 0.23
    CLα::Float64 = 5.61
    CLδ::Float64 = 0.13
    CM0::Float64 = 0.0135
    CMα::Float64 = -2.74
    CMδ::Float64 = -0.99
    CLq::Float64 = 7.95
    CMq::Float64 = -38.21

    CDt::Float64 = 0.8


    α_stall::Float64 = deg2rad(15.0) # stall angle of attack
    M_sigmoid::Float64 = 50.0

end

angle_normalize(θ) = mod2pi(θ+π) - π

fpa(x) = atan(x[5],x[4])
aoa(x, θ) = θ - fpa(x) |> angle_normalize # angle of attack
Va2(x) = x[4]^2 + x[5]^2 # airspeed squared

function σ_stall(α, α_stall, M_sigmoid)
    t1 = exp(-M_sigmoid*(α - α_stall))
    t2 = exp(M_sigmoid*(α + α_stall))
    return (1.0 + t1 + t2)/((1.0 + t1)*(1.0 + t2))
end

function M_aero_sim(u,x,p::VTOLParams)
    α = aoa(x, x[3])
    V2 = Va2(x)
    σ = σ_stall(α, p.α_stall, p.M_sigmoid)

    return 0.5*p.ρ*p.S*p.c*V2*(p.Cmδ*u[4] + (p.Cmα*α + p.Cm0)*(1-σ) + σ*-2*sign(α)*sin(α)^2*cos(α)) + 0.25*p.ρ*p.S*p.c^2*sqrt(V2)*p.CMq*x[6]
end

function F_aero_sim(u,x,p::VTOLParams)
    α = aoa(x, x[3])
    V2 = Va2(x)
    σ = σ_stall(α, p.α_stall, p.M_sigmoid)

    return 0.5*p.ρ*p.S*V2^2*[
                        -(p.CDδ*u[4] + p.CDα*α^2 + p.CD0 + p.CDt*(u[1] + u[2])); # drag force in wind X
                        p.CLδ*u[4] + (p.CLα*α + p.CL0)*(1-σ) + σ*2*sign(α)*sin(α)^2*cos(α); # lift force in wind Z
                        ] + [0.0; 0.25*p.ρ*p.S*p.c*sqrt(V2)*p.CLq*x[6]]
end

rear_prop_degradation(x,u) = u[3]^2*(1.0 - 0.5)*(1.0 + sqrt(Va2(x))/80.0)

M_prop_sim(x,u,p::VTOLParams) = p.l_motor*p.max_thrust_vert*(u[1]^2-u[2]^2*(1-rear_prop_degradation(x,u)))

F_prop_sim(x,u,p::VTOLParams) = @SVector[
  p.max_thrust_horz*u[3]^2; # body X
  p.max_thrust_vert*(u[1]^2 + u[2]^2*(1-rear_prop_degradation(x,u)))  # body Z
]

R(θ) = @SMatrix[ cos(θ) -sin(θ); 
         sin(θ)  cos(θ) ]


function dynamics_sim(x,u,p::VTOLParams)

    F_total = R(x[3])*F_prop_sim(x,u,p) + R(fpa(x))*F_aero_sim(u,x,p) + @SVector[0.0; -p.m*g]
    M_total = M_prop_sim(x,u,p) + M_aero_sim(u,x,p)

    dx = @SVector[
        x[4]; # xdot
        x[5]; # zdot
        x[6]; # θdot
        F_total[1]/p.m; # xddot
        F_total[2]/p.m; # zddot
        M_total/p.J; # θddot
    ]

    return dx
end

end