#![deny(warnings)]
use drg::attacks::{
    attack, attack_with_profile, AttackAlgo, AttackProfile, GreedyParams, TargetRange,
};
use drg::graph::{DRGAlgo, Graph, GraphSpec};
use drg::utils;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::fs::File;
use std::io::{self,Write};
use std::time::Instant;

use clap::{value_t_or_exit, App, Arg, ArgMatches, SubCommand};
#[cfg(feature = "cpu-profile")]
use gperftools::profiler::PROFILER;

const GRAPH_BENCH_CMD :&str = "graph-bench";
const ATTACK_CMD :&str = "attack";
const ATTACK_VALIANT :&str = "valiant";
const ATTACK_GREEDY :&str= "greedy";

const DRG_BUCKET :&str = "bucket";
const DRG_REN21 :&str = "ren21";

/// Start profile (currently use for the Greedy attack) and dump the file in
/// the current directory. It can later be analyzed with `pprof`, e.g.,
/// ```text
/// cargo run --release --features cpu-profile  -- -n 14 greedy
/// REV=$(git rev-parse --short HEAD)
/// pprof --lines --dot target/release/drg-attacks greedy.profile > profile-$REV.dot && xdot profile-$REV.dot &
/// ```
#[cfg(feature = "cpu-profile")]
#[inline(always)]
fn start_profile(stage: &str) {
    PROFILER
        .lock()
        .unwrap()
        .start(format!("./{}.profile", stage))
        .unwrap();
}
#[cfg(feature = "cpu-profile")]
#[inline(always)]
fn stop_profile() {
    PROFILER.lock().unwrap().stop().unwrap();
}
#[cfg(not(feature = "cpu-profile"))]
#[inline(always)]
fn start_profile(_stage: &str) {}
#[cfg(not(feature = "cpu-profile"))]
#[inline(always)]
fn stop_profile() {}

fn graph_bench(m: &ArgMatches) {
    let sub = m.subcommand_matches(GRAPH_BENCH_CMD).expect("subcommand graph not recognized");
    let pow = value_t_or_exit!(sub,"size",usize);
    let n = 1 << pow;
    let degree = value_t_or_exit!(sub, "degree", usize);
    let algo = match sub.value_of("drg").unwrap()  {
        DRG_BUCKET => DRGAlgo::MetaBucket(degree),
        DRG_REN21 => DRGAlgo::Ren21(degree),
        _ => panic!("DRG Algo unknown"),
    };
    let mut avg :f64 = 0.0;
    let runs = value_t_or_exit!(sub,"runs",usize);
    println!("Benchmark is starting for graphs {:?}",algo);
    for i in 0..runs {
        println!("Generating graph {}/{} ...",i, runs);
        let now = Instant::now();
        let seed = rand::thread_rng().gen::<[u8; 32]>();
        let specs = GraphSpec {
            size: n,
            seed: seed,
            algo: algo,
        };
        let mut rng = ChaChaRng::from_seed(specs.seed.clone());
        Graph::new_from_rng(specs, &mut rng);
        avg += now.elapsed().as_millis() as f64;
    }
    let avg = (avg / (runs as f64)) as u64;
    println!("Average of {}ms to generate graph", avg);
}

fn drg_command(m: &ArgMatches) {
    let sub = m
        .subcommand_matches(ATTACK_CMD)
        .expect("subcommand drg not recognized");
    let is_beta = sub.is_present("beta");
    let is_alpha = sub.is_present("alpha");
    if (is_beta ^ is_alpha) == false {
        panic!("alpha and beta can not be used at the same time");
    }

    // TODO check on validity of inputs?
    let pow = value_t_or_exit!(sub, "size", usize);
    let n = 1 << pow;
    let degree = value_t_or_exit!(sub, "degree", usize);
    let algo = match sub.value_of("drg").unwrap()  {
        DRG_BUCKET => DRGAlgo::MetaBucket(degree),
        DRG_REN21 => DRGAlgo::Ren21(degree),
        _ => panic!("DRG Algo unknown"),
    };
    let seed = rand::thread_rng().gen::<[u8; 32]>();
    let specs = GraphSpec {
        size: n,
        seed: seed,
        algo: algo,
    };
    let runs = 1;
    let attack_type = sub.value_of("attack").unwrap();
    let greedy_params = {
        let mut s = GreedyParams::standard(pow);
        let is_radius = sub.is_present("radius");
        let is_topk = sub.is_present("topk");
        let is_depth = sub.is_present("depth");
        let is_reset = sub.is_present("noreset");
        let is_greedy_params = is_radius || is_topk || is_depth || is_reset;
        if is_greedy_params && attack_type != "greedy" {
            panic!("greedy attack doesn't take any --radius or --topk flag");
        }
        if is_radius {
            s.radius = value_t_or_exit!(sub,"radius",usize);
        }
        if is_topk {
            s.k = value_t_or_exit!(sub,"topk",usize);
        }
        if is_depth {
            s.length = value_t_or_exit!(sub,"depth",usize);
        }
        s.reset = if is_reset { false } else { true };
        s
    };

    let parse_bounds = |default:&str| -> (f64,f64) {
        let default_v = value_t_or_exit!(sub, default, f64);
        let to = if sub.is_present("to") {
            value_t_or_exit!(sub,"to",f64)
        } else {
            value_t_or_exit!(sub, default, f64)
        };
        let interval = if sub.is_present("increment") {
            value_t_or_exit!(sub,"increment",f64)
        } else {
            (to - default_v) / 3.0
        };
        (to,interval)
    };
    let (attack, range) = if is_alpha {
        // alpha is the proportion of nodes we want to keep according to the DRG
        // definition and in this set of alpha*n nodes, we try to find the
        // longest path.
        let alpha = value_t_or_exit!(sub, "alpha", f64);
        let (max_alpha,interval) = parse_bounds("alpha");
        let exclusion_set = 1.0 - alpha;
        let set_size = (exclusion_set * n as f64) as usize;
        let max_exclusion_set = 1.0 - max_alpha;
        let range = TargetRange {
            // we reverse the two because alpha = 1 - set_size so we go from
            // lowest to highest
            // TODO really fix this ambivalent way in the code
            end: exclusion_set,
            start: max_exclusion_set,
            interval: interval,
        };
        // however, the attack works by finding a set S of size 1-alpha such that
        // when *removed* from the main graph, then the main graph has a longest
        // path of a certain depth beta.
        match attack_type {
            ATTACK_VALIANT => (AttackAlgo::ValiantSize(set_size), range),
            ATTACK_GREEDY => (AttackAlgo::GreedySize(set_size,greedy_params), range),
            _ => panic!("unknown type"),
        }
    } else {
        let beta = value_t_or_exit!(sub, "beta", f64);
        let (end,interval) = parse_bounds("beta");
        let beta_size = (beta * n as f64) as usize;
        let range = TargetRange {
            start: beta,
            end: end,
            interval: interval,
        };
        match attack_type {
            ATTACK_VALIANT => (AttackAlgo::ValiantDepth(beta_size), range),
            ATTACK_GREEDY =>  (AttackAlgo::GreedyDepth(beta_size,greedy_params), range),
            _ => panic!("unknown type"),
        }
    };
    let profile = AttackProfile {
        runs,
        range,
        attack,
    };

    println!("Running attacks on graph {}",specs);

    start_profile("drg");
    let results = attack_with_profile(specs, &profile);
    stop_profile();
    let handler : Box<dyn Write> = if sub.is_present("csv") {
        let fname = sub.value_of("csv").unwrap_or("results.csv");
        Box::new(File::create(fname).expect("opening output file failed"))
    } else {
        Box::new(io::stdout())
    };
    results.to_csv(handler).expect("failed to write to CSV");
}

fn porep_comparison() {
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 13;
    let size = (2 as usize).pow(n);
    println!("Comparison with porep short paper with n = {}", size);
    let deg = 6;
    let fname = format!("porep_n{}_d{}.json", n, deg);

    let mut g1 = Graph::load_or_create(&fname, size, random_bytes, DRGAlgo::MetaBucket(deg));
    //let mut g1 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(deg));

    let depth = (0.25 * (size as f32)) as usize;
    println!("{}", g1.stats());
    println!("Trial #1 with target depth = 0.25n = {}", depth);
    attack(&mut g1, AttackAlgo::ValiantDepth(depth));

    //let set_size = (0.30 * (size as f32)) as usize;
    //println!(
    //"Trial #2 with target size set = 0.30n = {} (G-S = 0.7n)",
    //set_size
    //);
    //attack(&mut g1, AttackAlgo::ValiantSize(set_size));

    //println!(
    //"Trial #3 with Valiant AB16, target depth = 0.25n = {}",
    //depth
    //);
    /*attack(&mut g1, AttackAlgo::ValiantAB16(depth));*/

    println!("Trial #4 with Greedy DRS, target depth = 0.25n = {}", depth);
    attack(
        &mut g1,
        AttackAlgo::GreedySize(
            depth,
            GreedyParams {
                k: GreedyParams::k_ratio(n as usize),
                radius: 5,
                length: 16,
                reset: true,
                iter_topk: true,
                ..GreedyParams::default()
            },
        ),
    );

    // Comparison with porep short paper with n = 1048576
    // graph stats: size=1048576, min parents=1, max children=26
    // Trial #1 with target depth = 0.25n = 262144
    // Attack with ValiantDepth(262144)
    //         -> size 344275 = 0.3283n
    //         -> depth(G-S) 234005 = 0.2232n
    //         -> time elapsed: 54.654373484s
    // Trial #2 with target size set = 0.30n = 314572 (G-S = 0.7n)
    // Attack with ValiantSize(314572)
    //         -> size 344275 = 0.3283n
    //         -> depth(G-S) 234005 = 0.2232n
    //         -> time elapsed: 36.29261127s
    // Trial #3 with Valiant AB16, target depth = 0.25n = 262144
    // Attack with ValiantAB16(262144)
    //         -> size 319204 = 0.3044n
    //         -> depth(G-S) 247292 = 0.2358n
    //         -> time elapsed: 97.742500864s

    // NOTE: AB16 seems slower and less performant than the ValiantDepth
}

fn greedy_attacks(n: usize) {
    println!("Greedy Attacks parameters");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as usize).pow(n as u32);
    let deg = 6;
    let target_size = (0.30 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };
    let runs = 10;
    //attack(&mut g1, AttackAlgo::ValiantDepth(depth));

    let greed_params = GreedyParams {
        k: 50,
        radius: 4,
        reset: true,
        // length influences the number of points taken from topk in one iteration
        // if it is too high, then too many nodes will be in the radius so we'll
        // only take the first entry in topk but not the rest (since they'll be in
        // the radius set)
        length: 8,
        iter_topk: true,
        use_degree: false,
        parallel: false,
    };

    let mut profile = AttackProfile::from_attack(
        AttackAlgo::GreedySize(target_size, greed_params.clone()),
        size,
    );
    profile.runs = runs;
    profile.range.start = 0.2;
    profile.range.end = 0.5;
    profile.range.interval = 0.1;

    start_profile("greedy");
    let res = attack_with_profile(spec, &profile);
    // FIXME: Turn this into a JSON output.
    println!("\n\n------------------");
    println!("Attack finished: {:?}", profile);
    stop_profile();
    let json = serde_json::to_string_pretty(&res).expect("can't serialize to json");
    println!("{}", json);
}

fn challenge_graphs() {
    let n_graphs = 5;
    let n = 20;
    let size = (2 as u32).pow(n);
    let degree = 6;
    (1..=n_graphs).for_each(|i| {
        let seed = rand::thread_rng().gen::<[u8; 32]>();
        let mut rng = ChaChaRng::from_seed(seed);
        let spec = GraphSpec {
            size: size as usize,
            seed: seed,
            algo: DRGAlgo::MetaBucket(degree),
        };
        println!(
            "Constructing graph with seed {}",
            utils::to_hex_string(&seed)
        );
        let g = Graph::new_from_rng(spec, &mut rng);
        let name = format!("graph-{}.json", i);
        let file = File::create(&name).unwrap();
        serde_json::to_writer(file, &g).unwrap();
        println!("\t-> saved to {}", name);
    });
}

fn baseline_valiant(n: usize) {
    println!("Baseline computation for target size [0.10,0.20,0.30]");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as usize).pow(n as u32);
    let deg = 6;
    let target_size = (0.30 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };

    // target depth
    let mut profile = AttackProfile::from_attack(AttackAlgo::ValiantDepth(target_size), size);
    profile.runs = 3;
    profile.range.start = 0.15;
    profile.range.end = 0.26;
    profile.range.interval = 0.05;

    let res1 = attack_with_profile(spec, &profile);
    // target size
    let mut profile = AttackProfile::from_attack(AttackAlgo::ValiantSize(target_size), size);
    profile.runs = 3;
    profile.range.start = 0.15;
    profile.range.end = 0.31;
    profile.range.interval = 0.05;

    let res2 = attack_with_profile(spec, &profile);
    let json = serde_json::to_string_pretty(&vec![res1, res2]).expect("can't serialize to json");
    println!("{}", json);
}

fn theoretical_limit() {
    let ts = 0.0000115;
    let td = 0.03;
    println!("Comparing against theoretical limit:");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    let deg = 6;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };

    let greed_params = GreedyParams {
        k: GreedyParams::k_ratio(n as usize),
        radius: 4,
        reset: true,
        length: 10,
        iter_topk: true,
        use_degree: true,
        parallel: true,
    };

    let mut profile = AttackProfile::from_attack(
        AttackAlgo::GreedySize((ts * size as f64) as usize, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.0000115;
    profile.range.end = 0.0000116;
    profile.range.interval = 0.1;

    let res0 = attack_with_profile(spec, &profile);
    println!(
        "json: {}",
        serde_json::to_string_pretty(&res0).expect("can't serialize to json")
    );

    let mut profile = AttackProfile::from_attack(
        AttackAlgo::GreedyDepth((td * size as f64) as usize, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.03;
    profile.range.end = 0.04;
    profile.range.interval = 0.01;

    let res1 = attack_with_profile(spec, &profile);
    println!(
        "json: {}",
        serde_json::to_string_pretty(&vec![res0, res1]).expect("can't serialize to json")
    );
}

fn baseline_greedy() {
    println!("Baseline computation for target size [0.10,0.20,0.30]");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    let deg = 6;
    let target_depth = (0.001 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };

    let greed_params = GreedyParams {
        k: GreedyParams::k_ratio(n as usize),
        radius: 4,
        reset: true,
        length: 10,
        iter_topk: true,
        use_degree: true,
        parallel: true,
    };

    let mut profile = AttackProfile::from_attack(
        AttackAlgo::GreedyDepth(target_depth, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.15;
    profile.range.end = 0.26;
    profile.range.interval = 0.05;

    let res1 = attack_with_profile(spec, &profile);

    let mut profile = AttackProfile::from_attack(
        AttackAlgo::GreedySize(target_depth, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.15;
    profile.range.end = 0.31;
    profile.range.interval = 0.05;
    let res2 = attack_with_profile(spec, &profile);

    let json = serde_json::to_string_pretty(&vec![res1, res2]).expect("can't serialize to json");
    println!("{}", json);
}

fn baseline_large() {
    println!("Baseline computation for target size [0.90]");
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    let deg = 6;
    let target_size = (0.30 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: DRGAlgo::MetaBucket(deg),
    };

    let greed_params = GreedyParams {
        k: GreedyParams::k_ratio(n as usize),
        radius: 4,
        reset: true,
        length: 10,
        iter_topk: true,
        use_degree: false,
        parallel: false,
    };

    let mut profile = AttackProfile::from_attack(
        AttackAlgo::GreedySize(target_size, greed_params.clone()),
        size,
    );
    profile.runs = 3;
    profile.range.start = 0.90;
    profile.range.end = 0.91;
    profile.range.interval = 0.10;

    let res = attack_with_profile(spec, &profile);
    println!("\n\n------------------");
    println!("Attack finished: {:?}", profile);
    let json = serde_json::to_string_pretty(&res).expect("can't serialize to json");
    println!("{}", json);
}

fn main() {
    pretty_env_logger::init_timed();
    let matches = App::new("DRG Attacks")
        .version("1.0")
        .arg(
            Arg::with_name("size")
                .short("n")
                .long("size-n")
                .help("Size of graph expressed as a power of 2")
                .default_value("10")
                .takes_value(true),
        )
        .subcommand(SubCommand::with_name(GRAPH_BENCH_CMD).about("Benchmarking graph generation")
            .arg(Arg::with_name("size")
                .long("size")
                .help("power of two of the size of the graph")
                .default_value("10")
                .takes_value(true)
            )
            .arg(Arg::with_name("drg")
                .long("drg")
                .default_value(DRG_BUCKET)
                .takes_value(true)
            )
            .arg(Arg::with_name("degree")
                .long("degree")
                .default_value("10")
                .takes_value(true)
            )
            .arg(Arg::with_name("runs")
                .long("runs")
                .default_value("3")
                .takes_value(true)
            )
        )
        .subcommand(SubCommand::with_name(ATTACK_CMD).about("general benchmark CLI to measure alphas of various configurations of DRGs")
            .arg(Arg::with_name("csv")
                .long("csv")
                .help("output file in CSV format")
            )
            .arg(Arg::with_name("drg")
                .long("drg")
                .help("Types of DRG graph")
                .default_value(DRG_BUCKET)
                .takes_value(true)
            )
            .arg(Arg::with_name("attack")
                .long("attack")
                .help("Type of attacks (valiant or greedy)")
                .default_value(ATTACK_VALIANT)
                .takes_value(true)
            )
            .arg(Arg::with_name("size")
                .short("n")
                .long("size")
                .help("Size of graph expressed as a power of 2")
                .default_value("10")
                .takes_value(true)
            ) 
            .arg(Arg::with_name("beta")
                .short("b")
                .long("beta")
                .help("Length of the longest path desired expressed in percentage of the graph size (i.e. 0.2). Attack will find exclusion set S such that depth(G-S) is inferior but the closest to beta")
                .takes_value(true)
            )
            .arg(Arg::with_name("alpha")
                .short("a")
                .long("alpha")
                .help("Size of the graph where we want to measure the longest path inside (definition of DRG).")
                .takes_value(true)
            )
            .arg(Arg::with_name("degree")
                .short("d")
                .long("degree")
                .help("Degree of nodes in the DRG")
                .required(true)
                .takes_value(true)
            ).arg(Arg::with_name("to")
                .long("to")
                .help("value of alpha/beta where to stop the attacks")
                .takes_value(true)
            ).arg(Arg::with_name("increment")
                .long("inc")
                .help("increments the value of alpha/beta each step")
                .default_value("0.1")
                .takes_value(true)
            )
            .arg(Arg::with_name("radius")
                .long("radius")
                .help("radius when attack is greedy: degree**radius/n should be small, i.e. 5-10%") 
                .takes_value(true)
            )
            .arg(Arg::with_name("topk")
                .long("topk")
                .help("number of nodes we select at each iteration of greedy")
                .takes_value(true)
            )
            .arg(Arg::with_name("depth")
                .long("depth")
                .help("maximum depth used by the greedy heuristic")
                .takes_value(true)
            )
            .arg(Arg::with_name("noreset")
                .long("noreset")
                .help("dont reset the inradius (default true)")
            )
        )
        .subcommand(SubCommand::with_name("greedy").about("Greedy attack"))
        .subcommand(SubCommand::with_name("challenge_graphs"))
        .subcommand(SubCommand::with_name("porep"))
        .subcommand(SubCommand::with_name("baseline_greedy"))
        .subcommand(SubCommand::with_name("baseline_valiant"))
        .subcommand(SubCommand::with_name("baseline_large"))
        .subcommand(SubCommand::with_name("theoretical_limit"))
        .get_matches();

    let n = value_t_or_exit!(matches, "size", usize);
    assert!(n < 50, "graph size is too big (2^{})", n);
    // FIXME: Use this argument for all attacks, not just Greedy (different
    // attacks may use different default values).

    if let Some(_) = matches.subcommand_matches("greedy") {
        greedy_attacks(n);
    } else if let Some(_) = matches.subcommand_matches("bounty") {
        challenge_graphs();
    } else if let Some(_) = matches.subcommand_matches("porep") {
        porep_comparison();
    } else if let Some(_) = matches.subcommand_matches("baseline_greedy") {
        baseline_greedy();
    } else if let Some(_) = matches.subcommand_matches("baseline_valiant") {
        baseline_valiant(n);
    } else if let Some(_) = matches.subcommand_matches("baseline_large") {
        baseline_large();
    } else if let Some(_) = matches.subcommand_matches("theoretical_limit") {
        theoretical_limit();
    } else if let Some(_) = matches.subcommand_matches(ATTACK_CMD) {
        drg_command(&matches);
    } else if let Some(_) = matches.subcommand_matches(GRAPH_BENCH_CMD) {
        graph_bench(&matches);
    } else {
        eprintln!("No subcommand entered, running `porep_comparison`");
        porep_comparison();
    }
    // FIXME: Can this be structured with a `match`?
}
