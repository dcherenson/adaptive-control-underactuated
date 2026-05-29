using Statistics
using DifferentialEquations, DiffEqCallbacks, StaticArrays

include("vtol.jl")
include("adaptation.jl")
include("high_level_control.jl")
include("control_allocation.jl")
include("simulation.jl")

import .ControlAllocation: n_x, n_u, n_λ, n_W
import .HighLevelController: ref_pose, ref_velocity
import .Simulation: PIDGains, CascadedPIDParams, SimulationParams, SimulationData, scenario_high_level_params, dudt!, cascaded_pid_allocator!
import .VTOL: VTOLParams

function metrics(pid::CascadedPIDParams; scenario=:landing, t_final=30.0)
    x0 = @MVector zeros(n_x)
    sim = SimulationParams(high_level=scenario_high_level_params(scenario), pid=pid, t_final=t_final)
    traj = sim.high_level.ref_traj
    r0 = ref_pose(sim.t0, traj)
    rdot0 = ref_velocity(sim.t0, traj)
    x0[1] = r0[1]; x0[4] = rdot0[1]
    x0[2] = 1.0 + r0[2]; x0[5] = rdot0[2]
    x0[3] = r0[3] + deg2rad(4.0)

    λ = @MVector zeros(n_λ)
    u = @MVector[0.01, 0.01, 0.4, -0.3, x0[3]]
    vtol = VTOLParams()
    W_true = @SVector[vtol.CDδ, vtol.CDα, vtol.CDt, vtol.CLδ, vtol.CLα, vtol.Cmδ, vtol.Cmα, 1.0, 0.0]
    W = MVector{n_W}(W_true*1.5)

    cb = PeriodicCallback(cascaded_pid_allocator!, sim.dt; initial_affect=true)
    data = SimulationData(u=copy(u), λ=copy(λ), W=copy(W), xhat=copy(x0), params=sim)
    prob = ODEProblem(dudt!, copy(x0), (sim.t0, sim.t_final), data)
    sol = solve(prob, Tsit5(), callback=cb, abstol=1e-6, reltol=1e-6)

    ex = Float64[]; ez = Float64[]; et = Float64[]
    for (i,t) in enumerate(data.t_hist)
        x=sol(t); r=ref_pose(t,traj)
        push!(ex, x[1]-r[1]); push!(ez, x[2]-r[2]); push!(et, x[3]-data.u_hist[i][5])
    end
    return sqrt(mean(abs2,ex)), sqrt(mean(abs2,ez)), rad2deg(sqrt(mean(abs2,et)))
end

base = CascadedPIDParams()
tuned = CascadedPIDParams(
    theta_hover=PIDGains(kp=6.228845113923736, ki=0.7627180213961051, kd=1.610646975999857, i_limit=0.21235433329717518, out_limit=2.607409204103294),
    theta_fw=PIDGains(kp=4.140902579583811, ki=0.21548281390889493, kd=0.19465227221502693, i_limit=0.12829925812585746, out_limit=0.996981866672961),
    q_hover=PIDGains(kp=12.540123852871549, ki=0.6534211743132547, kd=0.12632870115792455, i_limit=0.30452218869016134, out_limit=46.14207499618037),
    q_fw=PIDGains(kp=2.283473956525624, ki=0.7075176757989048, kd=0.04184332007633861, i_limit=0.40835642307919157, out_limit=60.0),
    hover_speed=8.519220520703417,
    fixed_wing_speed=10.519220520703417,
    moment_to_elevator_gain=-0.014645461382071314,
)

for sc in (:landing, :sinusoidal_transition)
    tf = sc == :landing ? 30.0 : 20.0
    bx,bz,bt = metrics(base; scenario=sc, t_final=tf)
    tx,tz,tt = metrics(tuned; scenario=sc, t_final=tf)
    println("scenario=$(sc)")
    println("  base  rmse_x=$(round(bx,digits=3)) rmse_z=$(round(bz,digits=3)) rmse_pitch_deg=$(round(bt,digits=3))")
    println("  tuned rmse_x=$(round(tx,digits=3)) rmse_z=$(round(tz,digits=3)) rmse_pitch_deg=$(round(tt,digits=3))")
end
