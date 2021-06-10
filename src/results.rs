use crate::attacks::AttackAlgo;
use crate::graph::{DRGAlgo, GraphSpec};
use csv;
use serde::{Deserialize, Serialize};
use std::io::Write;

/// Results of an attack expressed in relation to the graph size.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct SingleAttackResult {
    // Longest path found
    pub depth: f64,
    // Size of the set S removed such that the lnngest path found in the graph
    // (G - S) = depth.
    pub exclusion_size: f64,
    // graph_size: usize,
    // FIXME: Do we care to know the absolute number or just
    // relative to the graph size?
}

impl<'a> std::iter::Sum<&'a Self> for SingleAttackResult {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(SingleAttackResult::default(), |a, b| SingleAttackResult {
            depth: a.depth + b.depth,
            exclusion_size: a.exclusion_size + b.exclusion_size,
        })
    }
}

/// Average of many `SingleAttackResult`s.
// FIXME: Should be turn into a more generalized structure that also
//  has the variance along with the mean using a specialized crate.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct AveragedAttackResult {
    // for log/output purpose
    pub target: f64,     // target is what we ideally wanted to reach
    pub mean_depth: f64, // here are the actual values we reached
    pub mean_size: f64,
}

impl AveragedAttackResult {
    pub fn from_results(target: f64, results: &[SingleAttackResult]) -> Self {
        let aggregated: SingleAttackResult = results.iter().sum();
        AveragedAttackResult {
            mean_depth: aggregated.depth / results.len() as f64,
            mean_size: aggregated.exclusion_size / results.len() as f64,
            target: target,
        }
    }
}

/// Struct containing all informations about the attack runs. It can be
/// serialized into JSON or other format with serde.
#[derive(Debug, Serialize, Deserialize)]
pub struct Results {
    attacks: Vec<AttackResults>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AttackResults {
    pub spec: GraphSpec,
    // number of runs to average out the results
    pub runs: usize,
    pub attack: AttackAlgo,
    pub results: Vec<AveragedAttackResult>,
}

impl std::fmt::Display for SingleAttackResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "\t-> |S| = {:.2}\n\t-> depth(G-S) = {:.2}",
            self.exclusion_size, self.depth,
        )
    }
}

impl AttackResults {
    pub fn to_csv<W: Write>(&self, w: W) -> Result<(), csv::Error> {
        #[derive(Serialize)]
        struct Record {
            graph_size: u32,
            graph_type: String,
            degree: u32,
            attack_type: String,
            target_type: String,
            target: f64,
            alpha: f64,
            beta: f64,
        }
        let mut wtr = csv::Writer::from_writer(w);
        let n = self.spec.size as f64;
        let logn = n.log2() as u32;
        let (attack_type, target_type) = match self.attack {
            // transform alpha definition from block depth robust to DRG
            AttackAlgo::ValiantSize(_) => ("valiant", "alpha"),
            // beta stays the same
            AttackAlgo::ValiantDepth(_) => ("valiant", "beta"),
            AttackAlgo::GreedySize(_, _) => ("greedy", "alpha"),
            AttackAlgo::GreedyDepth(_, _) => ("greedy", "beta"),
            _ => panic!("unknown type of attack to serialize into csv"),
        };
        let (graph_type, degree) = match self.spec.algo {
            DRGAlgo::BucketSample => ("bucket", 2),
            DRGAlgo::MetaBucket(d) => ("meta-bucket", d),
            DRGAlgo::Ren21(d) => ("ren21", d),
            DRGAlgo::KConnector(k) => ("Kconnector", k),
        };
        let truncate = |before: f64| (before * 100.0).floor() / 100.0;
        self.results.iter().try_for_each(|r| {
            // mean size is the size of the set we remove
            let alpha = 1.0 - r.mean_size;
            let target = match target_type {
                "alpha" => 1.0 - r.target,
                _ => r.target,
            };
            wtr.serialize(Record {
                graph_size: logn,
                graph_type: graph_type.to_string(),
                degree: degree as u32,
                attack_type: attack_type.to_string(),
                target_type: target_type.to_string(),
                target: truncate(target),
                alpha: truncate(alpha),
                beta: truncate(r.mean_depth),
            })
        })
    }
}
