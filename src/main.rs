#![deny(warnings)]
mod attacks;
pub mod graph;
mod utils;
use attacks::{attack, attack_with_profile, AttackProfile, DepthReduceSet, GreedyParams};
use graph::{DRGAlgo, Graph, GraphSpec};
use rand::Rng;

#[macro_use]
#[cfg(test)]
extern crate lazy_static;

use clap::{value_t, App, Arg, SubCommand};
#[cfg(feature = "cpu-profile")]
use gperftools::profiler::PROFILER;

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

fn porep_comparison() {
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let n = 20;
    let size = (2 as usize).pow(n);
    println!("Comparison with porep short paper with n = {}", size);
    let deg = 6;
    let fname = format!("porep_n{}_d{}.json", n, deg);

    let mut g1 = Graph::load_or_create(&fname, size, random_bytes, DRGAlgo::MetaBucket(deg));
    //let mut g1 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(deg));

    let depth = (0.25 * (size as f32)) as usize;
    println!("{}", g1.stats());
    println!("Trial #1 with target depth = 0.25n = {}", depth);
    //attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

    //let set_size = (0.30 * (size as f32)) as usize;
    //println!(
    //"Trial #2 with target size set = 0.30n = {} (G-S = 0.7n)",
    //set_size
    //);
    //attack(&mut g1, DepthReduceSet::ValiantSize(set_size));

    //println!(
    //"Trial #3 with Valiant AB16, target depth = 0.25n = {}",
    //depth
    //);
    /*attack(&mut g1, DepthReduceSet::ValiantAB16(depth));*/

    println!("Trial #4 with Greedy DRS, target depth = 0.25n = {}", depth);
    attack(
        &mut g1,
        DepthReduceSet::GreedySize(
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
    //attack(&mut g1, DepthReduceSet::ValiantDepth(depth));

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
    };

    let mut profile = AttackProfile::from_attack(
        DepthReduceSet::GreedySize(target_size, greed_params.clone()),
        size,
    );
    // FIXME: Build the profile in one statement instead of making it mutable.
    profile.runs = 3;
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

fn baseline(k: usize, uniform_graph: bool, target_der: f32, runs: usize) {
    println!("Baseline computation for target size [0.10,0.20,0.30]");
    println!("Size of graph: 2^{}", k);
    let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
    let size = (2 as usize).pow(k as u32);
    let deg = 6;
    let target_size = (0.30 * size as f64) as usize;
    let spec = GraphSpec {
        size,
        seed: random_bytes,
        algo: if !uniform_graph {
            DRGAlgo::MetaBucket(deg)
        } else {
            DRGAlgo::UniformGraph {
                m: deg,
                ner: target_der.round() as usize,
            }
        },
    };

    let mut profile =
        AttackProfile::from_attack(DepthReduceSet::ExchangeNodes(target_size, target_der), size);
    profile.runs = runs;
    profile.range.start = 0.30;
    profile.range.end = 0.30;
    profile.range.interval = 0.10;
    // FIXME: Not enforcing max size at the moment, the attack naturally
    // stays close to `e = 0.1`.

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
            Arg::with_name("log-size")
                .short("k")
                .help("Size of graph expressed as a power of 2")
                .default_value("10")
                .takes_value(true),
        )
        .subcommand(SubCommand::with_name("greedy").about("Greedy attack"))
        .subcommand(SubCommand::with_name("porep"))
        .subcommand(
            SubCommand::with_name("baseline")
                .arg(
                    Arg::with_name("uniform-graph")
                        .short("u")
                        .help("use the uniform graph construction instead of the metabucket"),
                )
                .arg(
                    Arg::with_name("runs")
                        .long("runs")
                        .short("r")
                        .default_value("1")
                        .help("number of runs"),
                )
                .arg(
                    Arg::with_name("target-der")
                        .long("target-DER")
                        .short("d")
                        .default_value("4")
                        .help(
                        "total DER expected (actually applied to the NER part, not split for now)",
                    ),
                ),
        )
        .get_matches();

    let k = value_t!(matches, "log-size", usize).unwrap();
    assert!(k < 50, "graph size is too big (2^{})", k);
    // FIXME: Use this argument for all attacks, not just Greedy (different
    // attacks may use different default values).

    if let Some(_) = matches.subcommand_matches("greedy") {
        greedy_attacks(k);
    } else if let Some(_) = matches.subcommand_matches("porep") {
        porep_comparison();
    } else if let Some(matches) = matches.subcommand_matches("baseline") {
        baseline(
            k,
            matches.is_present("uniform-graph"),
            value_t!(matches.value_of("target-der"), f32).unwrap(),
            value_t!(matches.value_of("runs"), usize).unwrap(),
        );
    } else {
        eprintln!("No subcommand entered, running `porep_comparison`");
        porep_comparison();
    }
    // FIXME: Can this be structured with a `match`?
}
