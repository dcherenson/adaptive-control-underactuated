using Random, Statistics
using DifferentialEquations, DiffEqCallbacks, StaticArrays

include("vtol.jl")
include("adaptation.jl")
include("high_level_control.jl")
include("control_allocation.jl")
include("simulation.jl")

import .ControlAllocation: n_x, n_u, n_λ, n_W
import .HighLevelController: ref_pose, ref_velocity
import .Simulation: PIDGains, CascadedPIDParams, SimulationParams, SimulationData, scenario_high_level_params, dudt!, adaptive_control_allocator!, cascaded_pid_allocator!
import .VTOL: VTOLParams

const SCENARIOS = (
    (name=:landing, t_final=30.0, weight=0.75),
    (name=:takeoff, t_final=30.0, weight=0.25),
    (name=:sinusoidal_transition, t_final=20.0, weight=0.20),
)

Base.@kwdef struct ErrorMetrics
    rmse_x::Float64 = Inf
    rmse_z::Float64 = Inf
    rmse_pitch_deg::Float64 = Inf
    end_x::Float64 = Inf
    end_z::Float64 = Inf
    end_pitch_deg::Float64 = Inf
    max_x::Float64 = Inf
    max_z::Float64 = Inf
    max_pitch_deg::Float64 = Inf
end

function make_initial_state(sim::SimulationParams)
    x0 = @MVector zeros(n_x)
    traj = sim.high_level.ref_traj

    r0 = ref_pose(sim.t0, traj)
    rdot0 = ref_velocity(sim.t0, traj)
    x0[1] = r0[1]
    x0[4] = rdot0[1]
    x0[2] = 1.0 + r0[2]
    x0[5] = rdot0[2]
    x0[3] = r0[3] + deg2rad(4.0)

    return x0
end

function make_data(sim::SimulationParams, x0)
    λ = @MVector zeros(n_λ)
    u = @MVector[0.01, 0.01, 0.4, -0.3, x0[3]]
    vtol = VTOLParams()
    W_true = @SVector[vtol.CDδ, vtol.CDα, vtol.CDt, vtol.CLδ, vtol.CLα, vtol.Cmδ, vtol.Cmα, 1.0, 0.0]
    W = MVector{n_W}(W_true * 1.5)
    return SimulationData(u=copy(u), λ=copy(λ), W=copy(W), xhat=copy(x0), params=sim)
end

function run_case(mode::Symbol, pid::CascadedPIDParams, scenario::Symbol, t_final::Float64)
    sim = SimulationParams(high_level=scenario_high_level_params(scenario), pid=pid, t_final=t_final)
    x0 = make_initial_state(sim)
    data = make_data(sim, x0)
    cb_fun = mode == :adaptive ? adaptive_control_allocator! : cascaded_pid_allocator!
    cb = PeriodicCallback(cb_fun, sim.dt; initial_affect=true)
    prob = ODEProblem(dudt!, copy(x0), (sim.t0, sim.t_final), data)

    sol = solve(prob, Tsit5(), callback=cb, abstol=1e-6, reltol=1e-6)

    t_vec = data.t_hist
    if length(t_vec) < 20
        return ErrorMetrics()
    end

    ex = Float64[]
    ez = Float64[]
    ep_deg = Float64[]

    traj = sim.high_level.ref_traj
    for (i, t) in enumerate(t_vec)
        x = sol(t)
        r = ref_pose(t, traj)
        push!(ex, x[1] - r[1])
        push!(ez, x[2] - r[2])
        push!(ep_deg, rad2deg(x[3] - data.u_hist[i][5]))
    end

    if any(!isfinite, ex) || any(!isfinite, ez) || any(!isfinite, ep_deg)
        return ErrorMetrics()
    end

    return ErrorMetrics(
        rmse_x=sqrt(mean(abs2, ex)),
        rmse_z=sqrt(mean(abs2, ez)),
        rmse_pitch_deg=sqrt(mean(abs2, ep_deg)),
        end_x=abs(ex[end]),
        end_z=abs(ez[end]),
        end_pitch_deg=abs(ep_deg[end]),
        max_x=maximum(abs, ex),
        max_z=maximum(abs, ez),
        max_pitch_deg=maximum(abs, ep_deg),
    )
end

function scenario_cost(m::ErrorMetrics, a::ErrorMetrics)
    # Core tracking quality
    c = 1.5*m.rmse_x + 2.0*m.rmse_z + 0.15*m.rmse_pitch_deg

    # End-of-horizon bias (important for landing)
    c += 0.45*m.end_x + 0.80*m.end_z + 0.20*m.end_pitch_deg

    # Large transient penalties
    c += 0.25*max(0.0, m.max_x - 8.0)^2
    c += 0.35*max(0.0, m.max_z - 6.0)^2
    c += 0.20*max(0.0, m.max_pitch_deg - 15.0)^2

    # Feasibility guardrails for terminal behavior
    c += 1.2*max(0.0, m.end_x - 15.0)^2
    c += 2.0*max(0.0, m.end_z - 2.0)^2
    c += 0.8*max(0.0, m.end_pitch_deg - 6.0)^2

    # Comparability-to-adaptive penalties (floored so adaptive~0 does not explode)
    target_x = max(3.0 * a.rmse_x, 3.0)
    target_z = max(3.0 * a.rmse_z, 2.0)
    target_p = max(3.0 * a.rmse_pitch_deg, 6.0)

    c += 4.5 * max(0.0, m.rmse_x - target_x)^2
    c += 5.5 * max(0.0, m.rmse_z - target_z)^2
    c += 0.8 * max(0.0, m.rmse_pitch_deg - target_p)^2

    return c
end

function total_cost(pid::CascadedPIDParams, adaptive_baseline, active_scenarios)
    total = 0.0
    metrics = Dict{Symbol, ErrorMetrics}()

    for sc in active_scenarios
        m = run_case(:pid, pid, sc.name, sc.t_final)
        a = adaptive_baseline[sc.name]
        metrics[sc.name] = m
        total += sc.weight * scenario_cost(m, a)
    end

    return total, metrics
end

function clamp_pid(pid::CascadedPIDParams)
    hs = clamp(pid.hover_speed, 3.0, 12.0)
    fs = clamp(pid.fixed_wing_speed, hs + 1.0, 22.0)

    return CascadedPIDParams(
        theta_hover=PIDGains(
            kp=clamp(pid.theta_hover.kp, 1.0, 12.0),
            ki=clamp(pid.theta_hover.ki, 0.0, 3.0),
            kd=clamp(pid.theta_hover.kd, 0.0, 3.0),
            i_limit=clamp(pid.theta_hover.i_limit, 0.05, 1.0),
            out_limit=clamp(pid.theta_hover.out_limit, 0.3, 6.0),
        ),
        theta_fw=PIDGains(
            kp=clamp(pid.theta_fw.kp, 0.5, 10.0),
            ki=clamp(pid.theta_fw.ki, 0.0, 3.0),
            kd=clamp(pid.theta_fw.kd, 0.0, 2.0),
            i_limit=clamp(pid.theta_fw.i_limit, 0.05, 1.0),
            out_limit=clamp(pid.theta_fw.out_limit, 0.2, 5.0),
        ),
        q_hover=PIDGains(
            kp=clamp(pid.q_hover.kp, 2.0, 20.0),
            ki=clamp(pid.q_hover.ki, 0.0, 6.0),
            kd=clamp(pid.q_hover.kd, 0.0, 2.0),
            i_limit=clamp(pid.q_hover.i_limit, 0.05, 2.0),
            out_limit=clamp(pid.q_hover.out_limit, 8.0, 100.0),
        ),
        q_fw=PIDGains(
            kp=clamp(pid.q_fw.kp, 1.0, 16.0),
            ki=clamp(pid.q_fw.ki, 0.0, 6.0),
            kd=clamp(pid.q_fw.kd, 0.0, 2.0),
            i_limit=clamp(pid.q_fw.i_limit, 0.05, 2.0),
            out_limit=clamp(pid.q_fw.out_limit, 8.0, 100.0),
        ),
        hover_speed=hs,
        fixed_wing_speed=fs,
        moment_to_elevator_gain=clamp(pid.moment_to_elevator_gain, -0.08, -0.002),
        fw_ref_pitch_weight=clamp(pid.fw_ref_pitch_weight, 0.0, 1.0),
        fw_accel_to_pitch_gain=clamp(pid.fw_accel_to_pitch_gain, -0.4, 0.4),
        ax_to_elevator_gain=clamp(pid.ax_to_elevator_gain, -0.08, 0.08),
        u_rate_limits=pid.u_rate_limits,
        throttle_sum_limit=clamp(pid.throttle_sum_limit, 1.2, 2.2),
    )
end

function perturb(base::CascadedPIDParams, rng::AbstractRNG, scale::Float64)
    # Multiplicative noise for positive gains, additive for signed/speed terms.
    sf() = exp(randn(rng) * scale)

    p = CascadedPIDParams(
        theta_hover=PIDGains(
            kp=base.theta_hover.kp * sf(),
            ki=base.theta_hover.ki * sf(),
            kd=base.theta_hover.kd * sf(),
            i_limit=base.theta_hover.i_limit * sf(),
            out_limit=base.theta_hover.out_limit * sf(),
        ),
        theta_fw=PIDGains(
            kp=base.theta_fw.kp * sf(),
            ki=base.theta_fw.ki * sf(),
            kd=base.theta_fw.kd * sf(),
            i_limit=base.theta_fw.i_limit * sf(),
            out_limit=base.theta_fw.out_limit * sf(),
        ),
        q_hover=PIDGains(
            kp=base.q_hover.kp * sf(),
            ki=base.q_hover.ki * sf(),
            kd=base.q_hover.kd * sf(),
            i_limit=base.q_hover.i_limit * sf(),
            out_limit=base.q_hover.out_limit * sf(),
        ),
        q_fw=PIDGains(
            kp=base.q_fw.kp * sf(),
            ki=base.q_fw.ki * sf(),
            kd=base.q_fw.kd * sf(),
            i_limit=base.q_fw.i_limit * sf(),
            out_limit=base.q_fw.out_limit * sf(),
        ),
        hover_speed=base.hover_speed + randn(rng) * (1.2 * scale / 0.25),
        fixed_wing_speed=base.fixed_wing_speed + randn(rng) * (1.4 * scale / 0.25),
        moment_to_elevator_gain=base.moment_to_elevator_gain * exp(randn(rng) * (0.7 * scale)),
        fw_ref_pitch_weight=base.fw_ref_pitch_weight + randn(rng) * (0.25 * scale / 0.25),
        fw_accel_to_pitch_gain=base.fw_accel_to_pitch_gain * exp(randn(rng) * (0.7 * scale)),
        ax_to_elevator_gain=base.ax_to_elevator_gain + randn(rng) * (0.02 * scale / 0.25),
        u_rate_limits=base.u_rate_limits,
        throttle_sum_limit=base.throttle_sum_limit + randn(rng) * (0.15 * scale / 0.25),
    )

    return clamp_pid(p)
end

function random_pid(rng::AbstractRNG)
    p = CascadedPIDParams(
        theta_hover=PIDGains(kp=rand(rng, 1.0:0.1:10.0), ki=rand(rng)*2.0, kd=rand(rng)*2.5, i_limit=0.1 + rand(rng)*0.7, out_limit=0.6 + rand(rng)*4.0),
        theta_fw=PIDGains(kp=rand(rng, 0.8:0.1:8.0), ki=rand(rng)*1.5, kd=rand(rng)*1.5, i_limit=0.08 + rand(rng)*0.6, out_limit=0.4 + rand(rng)*2.8),
        q_hover=PIDGains(kp=rand(rng, 3.0:0.2:18.0), ki=rand(rng)*4.0, kd=rand(rng)*0.9, i_limit=0.1 + rand(rng)*1.2, out_limit=15.0 + rand(rng)*65.0),
        q_fw=PIDGains(kp=rand(rng, 1.5:0.2:12.0), ki=rand(rng)*3.0, kd=rand(rng)*0.8, i_limit=0.1 + rand(rng)*1.0, out_limit=12.0 + rand(rng)*60.0),
        hover_speed=4.0 + rand(rng)*7.0,
        fixed_wing_speed=8.0 + rand(rng)*10.0,
        moment_to_elevator_gain=-(0.004 + rand(rng)*0.05),
        fw_ref_pitch_weight=rand(rng),
        fw_accel_to_pitch_gain=-0.2 + rand(rng)*0.4,
        ax_to_elevator_gain=-0.05 + rand(rng)*0.10,
        throttle_sum_limit=1.4 + rand(rng)*0.7,
    )
    return clamp_pid(p)
end

function fmt_metrics(m::ErrorMetrics)
    return "rmse[x=$(round(m.rmse_x,digits=2)), z=$(round(m.rmse_z,digits=2)), pitch=$(round(m.rmse_pitch_deg,digits=2))deg], end[x=$(round(m.end_x,digits=2)), z=$(round(m.end_z,digits=2)), pitch=$(round(m.end_pitch_deg,digits=2))deg]"
end

function main(; iterations=180, seed=13)
    rng = MersenneTwister(seed)

    println("Computing adaptive baselines...")
    adaptive_baseline = Dict{Symbol, ErrorMetrics}()
    active_scenarios = []
    for sc in SCENARIOS
        m = run_case(:adaptive, CascadedPIDParams(), sc.name, sc.t_final)
        if isfinite(m.rmse_x) && isfinite(m.rmse_z) && isfinite(m.rmse_pitch_deg)
            adaptive_baseline[sc.name] = m
            push!(active_scenarios, sc)
            println("adaptive $(sc.name): $(fmt_metrics(m))")
        else
            println("adaptive $(sc.name): unstable -> skipped")
        end
    end

    if isempty(active_scenarios)
        error("No stable adaptive-baseline scenarios available for tuning.")
    end

    # Candidate seeds: original defaults + current defaults + randoms
    seed_default_old = CascadedPIDParams(
        theta_hover=PIDGains(kp=3.5, ki=0.8, kd=0.6, i_limit=0.35, out_limit=2.0),
        theta_fw=PIDGains(kp=2.0, ki=0.35, kd=0.3, i_limit=0.25, out_limit=1.2),
        q_hover=PIDGains(kp=7.0, ki=1.2, kd=0.2, i_limit=0.5, out_limit=45.0),
        q_fw=PIDGains(kp=4.5, ki=0.7, kd=0.1, i_limit=0.35, out_limit=30.0),
        hover_speed=7.0,
        fixed_wing_speed=16.0,
        moment_to_elevator_gain=-0.02,
        fw_ref_pitch_weight=0.40,
        fw_accel_to_pitch_gain=-0.12,
        ax_to_elevator_gain=-0.02,
        throttle_sum_limit=1.95,
    )

    seeds = CascadedPIDParams[seed_default_old, CascadedPIDParams()]
    for _ in 1:6
        push!(seeds, random_pid(rng))
    end

    best = seeds[1]
    best_cost = Inf
    best_metrics = Dict{Symbol, ErrorMetrics}()

    for s in seeds
        c, ms = total_cost(s, adaptive_baseline, active_scenarios)
        if c < best_cost
            best = s
            best_cost = c
            best_metrics = ms
        end
    end

    println("Initial best cost=$(round(best_cost,digits=4))")

    # Evolutionary local/global search
    elites = [(best_cost, best)]

    for i in 1:iterations
        t = (i - 1) / max(1, iterations - 1)
        scale = 0.28 * (1 - t) + 0.08 * t

        parent = rand(rng) < 0.75 ? elites[rand(rng, 1:length(elites))][2] : best
        cand = if rand(rng) < 0.15
            random_pid(rng)
        else
            perturb(parent, rng, scale)
        end

        c = Inf
        ms = Dict{Symbol, ErrorMetrics}()
        try
            c, ms = total_cost(cand, adaptive_baseline, active_scenarios)
        catch
            c = Inf
        end

        if c < best_cost
            best = cand
            best_cost = c
            best_metrics = ms
            println("iter=$i improved cost=$(round(best_cost,digits=4))")
            for sc in active_scenarios
                println("  pid $(sc.name): $(fmt_metrics(best_metrics[sc.name]))")
            end
        elseif i % 20 == 0
            println("iter=$i no-improve current=$(round(best_cost,digits=4))")
        end

        if isfinite(c)
            push!(elites, (c, cand))
            sort!(elites, by=x -> x[1])
            if length(elites) > 8
                pop!(elites)
            end
        end
    end

    println("\n=== BEST PID ===")
    println(best)
    println("best_cost=$(best_cost)")
    for sc in active_scenarios
        println("adaptive $(sc.name): $(fmt_metrics(adaptive_baseline[sc.name]))")
        println("pid      $(sc.name): $(fmt_metrics(best_metrics[sc.name]))")
    end
end

main()
