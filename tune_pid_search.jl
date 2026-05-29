using Random, Statistics
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

function run_cost(pid::CascadedPIDParams; scenario=:landing, t_final=30.0)
    x0 = @MVector zeros(n_x)
    sim = SimulationParams(high_level=scenario_high_level_params(scenario), pid=pid, t_final=t_final)
    traj = sim.high_level.ref_traj

    r0 = ref_pose(sim.t0, traj)
    rdot0 = ref_velocity(sim.t0, traj)
    x0[1] = r0[1]
    x0[4] = rdot0[1]
    x0[2] = 1.0 + r0[2]
    x0[5] = rdot0[2]
    x0[3] = r0[3] + deg2rad(4.0)

    xhat0 = copy(x0)
    λ = @MVector zeros(n_λ)
    u = @MVector[0.01, 0.01, 0.4, -0.3, x0[3]]
    vtol = VTOLParams()
    W_true = @SVector[vtol.CDδ, vtol.CDα, vtol.CDt, vtol.CLδ, vtol.CLα, vtol.Cmδ, vtol.Cmα, 1.0, 0.0]
    W = MVector{n_W}(W_true*1.5)

    cb = PeriodicCallback(cascaded_pid_allocator!, sim.dt; initial_affect=true)
    data = SimulationData(u=copy(u), λ=copy(λ), W=copy(W), xhat=copy(xhat0), params=sim)
    prob = ODEProblem(dudt!, copy(x0), (sim.t0, sim.t_final), data)

    sol = solve(prob, Tsit5(), callback=cb, abstol=1e-6, reltol=1e-6)

    t_vec = data.t_hist
    if length(t_vec) < 20
        return Inf
    end

    err_x = Float64[]
    err_z = Float64[]
    err_theta = Float64[]
    u_use = Float64[]

    for (i, t) in enumerate(t_vec)
        x = sol(t)
        r = ref_pose(t, traj)
        push!(err_x, x[1] - r[1])
        push!(err_z, x[2] - r[2])
        push!(err_theta, x[3] - data.u_hist[i][5])
        uh = data.u_hist[i]
        push!(u_use, uh[1]^2 + uh[2]^2 + uh[3]^2 + 0.2*uh[4]^2)
    end

    if any(!isfinite, err_x) || any(!isfinite, err_z) || any(!isfinite, err_theta)
        return Inf
    end

    rmse_x = sqrt(mean(abs2, err_x))
    rmse_z = sqrt(mean(abs2, err_z))
    rmse_theta_deg = rad2deg(sqrt(mean(abs2, err_theta)))
    effort = mean(u_use)

    return 1.0*rmse_x + 1.4*rmse_z + 0.08*rmse_theta_deg + 0.03*effort
end

function tweak(base::CascadedPIDParams, rng::AbstractRNG)
    sf() = exp(rand(rng) * log(2.0) - log(sqrt(2.0)))

    th_h = PIDGains(
        kp=clamp(base.theta_hover.kp*sf(), 1.0, 8.0),
        ki=clamp(base.theta_hover.ki*sf(), 0.1, 2.0),
        kd=clamp(base.theta_hover.kd*sf(), 0.05, 2.0),
        i_limit=clamp(base.theta_hover.i_limit*sf(), 0.15, 1.0),
        out_limit=clamp(base.theta_hover.out_limit*sf(), 0.5, 4.0),
    )
    th_fw = PIDGains(
        kp=clamp(base.theta_fw.kp*sf(), 0.8, 6.0),
        ki=clamp(base.theta_fw.ki*sf(), 0.05, 1.5),
        kd=clamp(base.theta_fw.kd*sf(), 0.03, 1.5),
        i_limit=clamp(base.theta_fw.i_limit*sf(), 0.1, 0.8),
        out_limit=clamp(base.theta_fw.out_limit*sf(), 0.3, 3.5),
    )
    q_h = PIDGains(
        kp=clamp(base.q_hover.kp*sf(), 2.0, 14.0),
        ki=clamp(base.q_hover.ki*sf(), 0.1, 3.0),
        kd=clamp(base.q_hover.kd*sf(), 0.03, 1.5),
        i_limit=clamp(base.q_hover.i_limit*sf(), 0.1, 1.5),
        out_limit=clamp(base.q_hover.out_limit*sf(), 10.0, 80.0),
    )
    q_fw = PIDGains(
        kp=clamp(base.q_fw.kp*sf(), 1.0, 10.0),
        ki=clamp(base.q_fw.ki*sf(), 0.05, 2.5),
        kd=clamp(base.q_fw.kd*sf(), 0.02, 1.2),
        i_limit=clamp(base.q_fw.i_limit*sf(), 0.08, 1.0),
        out_limit=clamp(base.q_fw.out_limit*sf(), 8.0, 60.0),
    )

    hs = clamp(base.hover_speed + randn(rng)*0.8, 4.0, 11.0)
    fs = clamp(base.fixed_wing_speed + randn(rng)*1.0, hs + 2.0, 22.0)
    meg = clamp(base.moment_to_elevator_gain * exp(randn(rng)*0.18), -0.08, -0.004)

    return CascadedPIDParams(
        theta_hover=th_h,
        theta_fw=th_fw,
        q_hover=q_h,
        q_fw=q_fw,
        hover_speed=hs,
        fixed_wing_speed=fs,
        moment_to_elevator_gain=meg,
        u_rate_limits=base.u_rate_limits,
        throttle_sum_limit=base.throttle_sum_limit,
    )
end

function score(pid)
    c1 = run_cost(pid; scenario=:landing, t_final=30.0)
    c2 = run_cost(pid; scenario=:sinusoidal_transition, t_final=20.0)
    return 0.65*c1 + 0.35*c2, c1, c2
end

function main()
    rng = MersenneTwister(7)
    base = CascadedPIDParams()
    best = base
    best_total, best_landing, best_sine = score(base)
    println("baseline total=$(round(best_total,digits=4)) landing=$(round(best_landing,digits=4)) sine=$(round(best_sine,digits=4))")

    for i in 1:45
        cand = tweak(best, rng)
        total = Inf
        land = Inf
        sine = Inf
        try
            total, land, sine = score(cand)
        catch
            total = Inf
        end

        if total < best_total
            best = cand
            best_total = total
            best_landing = land
            best_sine = sine
            println("iter=$i improved total=$(round(total,digits=4)) landing=$(round(land,digits=4)) sine=$(round(sine,digits=4))")
        elseif i % 10 == 0
            println("iter=$i no-improve current=$(round(best_total,digits=4))")
        end
    end

    println("\nBEST")
    println(best)
    println("best_total=$(best_total) landing=$(best_landing) sine=$(best_sine)")
end

main()
