use std::cmp::{Ordering, Reverse};
use std::time::Instant;

use log::{debug, trace};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::graph::{EdgeSet, ExclusionSet, Graph, GraphSpec, Node, NodeSet};
use crate::results::{AttackResults, AveragedAttackResult, SingleAttackResult};
use crate::utils;
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttackAlgo {
    /// depth of the resulting G-S graph desired
    ValiantDepth(usize),
    /// size of the resulting exclusion set S desired - the size of the graph
    /// that we REMOVE from the original graph where we compute the final depth.
    ValiantSize(usize),
    /// AB16 Lemma 6.2 variant of the Valiant Lemma's based attack.
    /// Parameter  is size of the resulting G-S graph desired
    ValiantAB16(usize),
    /// depth of the resulting G-S graph and some specific parameters
    GreedyDepth(usize, GreedyParams),
    /// Variation of Greedy attack that has as `target` the resulting size of set S.
    GreedySize(usize, GreedyParams),
}

pub fn depth_reduce(g: &mut Graph, drs: AttackAlgo) -> ExclusionSet {
    match drs {
        AttackAlgo::ValiantDepth(_) => valiant_reduce(g, drs),
        AttackAlgo::ValiantSize(_) => valiant_reduce(g, drs),
        AttackAlgo::ValiantAB16(_) => valiant_reduce(g, drs),
        AttackAlgo::GreedyDepth(_, _) => greedy_reduce(g, drs),
        AttackAlgo::GreedySize(_, _) => greedy_reduce(g, drs),
    }
}

impl fmt::Display for AttackAlgo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttackAlgo::ValiantDepth(s) => write!(f, "ValiantDepth: path length {}", s),
            AttackAlgo::ValiantSize(s) => write!(f, "ValiantSize: exclusion set size {}", s),
            AttackAlgo::ValiantAB16(s) => write!(f, "ValiantAB16: graph remaining target {}", s),
            AttackAlgo::GreedyDepth(s, _) => write!(f, "GreedyDepth: path length {}", s),
            AttackAlgo::GreedySize(s, _) => write!(f, "GreedySize: exclusion set size {}", s),
        }
    }
}

/// Range of targets to try (to find the optimum value) from `start`, increasing
/// by `interval` until `end` is reached or surpassed.
// FIXME: Using this instead of `std::ops::Range<f64>` because Rust correctly
//  doesn't allow iterating over floating point values but there is probably
//  an easier way than coding this from scratch.
// FIXME: Assert range validity in the struct itself instead of on the caller
//  (`attack`).
#[derive(Debug)]
pub struct TargetRange {
    pub start: f64,
    pub interval: f64,
    pub end: f64,
}

#[derive(Debug)]
pub struct AttackProfile {
    pub runs: usize,
    pub range: TargetRange,
    pub attack: AttackAlgo,
}

impl AttackProfile {
    /// Build a default profile from an attack type that has only one run
    /// in a range of a single value (to make it compatible with previous
    /// uses of `attack`).
    // FIXME: We shouldn't need the `graph_size` (or the graph for that
    // matter), but this is accommodating previous uses of `attack` (which
    // should be refactored entirely and this method removed or reworked).
    pub fn from_attack(attack: AttackAlgo, graph_size: usize) -> Self {
        let graph_size = graph_size as f64;
        let target = match attack {
            AttackAlgo::ValiantDepth(depth) => depth as f64 / graph_size,
            AttackAlgo::ValiantSize(size) => size as f64 / graph_size,
            AttackAlgo::ValiantAB16(size) => size as f64 / graph_size,
            AttackAlgo::GreedyDepth(depth, _) => depth as f64 / graph_size,
            AttackAlgo::GreedySize(size, _) => size as f64 / graph_size,
        };
        // FIXME: This code should absorb the `depth_reduce` and derived
        // functions logic. The target discrimination depth/size should
        // b independent of the attack type (valiant/greedy).

        let range = {
            // FIXME: Too verbose, there probably is a more concise way to do this.
            TargetRange {
                start: target,
                end: target,
                interval: 0.0,
            }
        };

        AttackProfile {
            runs: 1,
            range,
            attack,
        }
    }
}

pub fn attack(g: &mut Graph, attack: AttackAlgo) -> SingleAttackResult {
    let start = Instant::now();
    let set = depth_reduce(g, attack);
    let duration = start.elapsed();
    let depth = g.depth_exclude(&set);
    let result = SingleAttackResult {
        depth: depth as f64 / g.size() as f64,
        exclusion_size: set.size() as f64 / g.size() as f64,
    };
    println!("{}", result);
    println!("\t-> time elapsed: {:?}", duration);
    result
}

// FIXME: Eventually this should replace the old `attack`.
pub fn attack_with_profile(spec: GraphSpec, profile: &AttackProfile) -> AttackResults {
    let mut targets: Vec<f64> = Vec::new();
    let mut target = profile.range.start;
    loop {
        targets.push(target);
        target += profile.range.interval;
        if target >= profile.range.end {
            break;
        }
    }
    // FIXME: Move this logic to `TargetRange`.

    let mut results: Vec<Vec<SingleAttackResult>> =
        vec![vec![SingleAttackResult::default(); profile.runs]; targets.len()];

    // Iterate over the graphs first (that means iterating over each run in
    // the outer `for`) to avoid memory bloat, we don't need to retain a
    // graph once we attacked it with all targets.
    let mut rng = ChaCha20Rng::from_seed(spec.seed);
    for run in 0..profile.runs {
        println!(
            "Attack {}:\n\t-> Graph generation with spec {}",
            run,
            spec.clone()
        );
        let mut g = Graph::new_from_rng(spec, &mut rng);

        for (t, target) in targets.iter().enumerate() {
            let absolute_target = (target * spec.size as f64) as usize;
            let attack_type = match profile.attack.clone() {
                AttackAlgo::ValiantDepth(_) => AttackAlgo::ValiantDepth(absolute_target),
                AttackAlgo::ValiantSize(_) => AttackAlgo::ValiantSize(absolute_target),
                AttackAlgo::ValiantAB16(_) => AttackAlgo::ValiantAB16(absolute_target),
                AttackAlgo::GreedyDepth(_, p) => AttackAlgo::GreedyDepth(absolute_target, p),
                AttackAlgo::GreedySize(_, p) => AttackAlgo::GreedySize(absolute_target, p),
            };
            println!("\t-> Run target attack {:.2}, with {}", attack_type, target);
            results[t][run] = attack(&mut g, attack_type.clone());
        }
    }

    AttackResults {
        spec: spec,
        runs: profile.runs,
        attack: profile.attack.clone(),
        results: targets
            .iter()
            .enumerate()
            .map(|(i, &target)| AveragedAttackResult::from_results(target, &results[i]))
            .collect(),
    }
}

// GreedyParams holds the different parameters to choose for the greedy algorithm
// such as the radius from which to delete nodes and the heuristic length.
// NOTE: degree**radius/size should be small, i.e. if it's too big the
// radiuis set will quickly contains all the nodes so there won't be any new
// nodesto add to the exclusion set (we don't take nodes included radius to the
// exclusion set). 5-30% should be fine.
// The topk parameters determines how much steps roughy the algorithm will take
// numbers of steps ~= size / topk  (assuming the radius doesn't take every
// nodes at some points in the computation, at which point we will include much
// less than topk nodes at each iteration; this would happen once the G-S
// becomes small).
// TODO: the topk parameters and radius should evolve gradually through time and
// not be fixed. In particular, radius should become smaller and smaller as
// G-S becomes smaller and smaller. If topk is small enough it doesn't need to
// change, but it should never be a high proportion of the graph size (50%)
// because including 50% of nodes in one step will give less precise results as
// well. Option to explore as this is likely to give better results but worse
// performance as well.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GreedyParams {
    // how many k nodes do we "remove" at each iteration in append_removal
    pub k: usize,
    // the radius for the heuristic to delete as well close nodes within a
    // radius of a selected node.
    pub radius: usize,
    // maximum lenth of the path - heuristic for the table produced by count_paths
    // see paragraph below equation 8.
    pub length: usize,
    // use parallelism for certain parts of the attacks
    pub parallel: bool,
    // test field to look at the impact of reseting the inradius set between
    // iterations or not.
    // FIXME: it can sometimes create an infinite loop
    // depending on the graph: if the inradius set contains the whole graph,
    // the greedy_reduce will loop infinitely
    pub reset: bool,
    // test field: when set, the topk nodes are selected one by one, updating the
    // radius set for each selected node.
    pub iter_topk: bool,
    // when set to true, greedy counts the degree of a node as
    // an indicator of its number of incident path
    pub use_degree: bool,
}

impl GreedyParams {
    // TODO find better formulas/rational
    pub fn k_ratio(log2n: usize) -> usize {
        if log2n >= 20 {
            (2 as usize).pow((log2n as u32 - 18) / 2) * 400
        } else {
            (1 << log2n) / 10
        }
    }
    pub fn standard(size: usize) -> Self {
        GreedyParams {
            k: Self::k_ratio(size),
            radius: 8,
            length: 16,
            reset: true,
            iter_topk: true,
            parallel: true,
            ..GreedyParams::default()
        }
    }
}

// greedy_reduce implements the Algorithm 5 of https://eprint.iacr.org/2018/944.pdf
fn greedy_reduce(g: &mut Graph, d: AttackAlgo) -> ExclusionSet {
    match d {
        AttackAlgo::GreedyDepth(depth, p) => {
            greedy_reduce_main(g, p, &|set: &ExclusionSet, g: &mut Graph| {
                g.depth_exclude(set) > depth
            })
        }
        AttackAlgo::GreedySize(size, p) => {
            // FIXME: To hit exactly the `target_size` we should consider the number of nodes
            //  removed in each iteration (`GreedyParams::k`), but since that number is small
            //  compared to normal target sizes it is an acceptable bias for now. We only
            //  correct `k` if it's bigger than 1/100th the target size.
            let mut p = p.clone();
            p.k = std::cmp::min(p.k, (size as f32 * 0.01).ceil() as usize);

            greedy_reduce_main(g, p, &|set: &ExclusionSet, _: &mut Graph| set.size() < size)
        }
        _ => panic!("invalid AttackAlgo option"),
    }
}

fn greedy_reduce_main(
    g: &mut Graph,
    p: GreedyParams,
    f: &dyn Fn(&ExclusionSet, &mut Graph) -> bool,
) -> ExclusionSet {
    let mut s = ExclusionSet::new(g);
    g.children_project();
    let mut inradius: NodeSet = NodeSet::default();
    while f(&s, g) {
        // TODO use p.length when more confidence in the trick
        let incidents = count_paths(g, &s, &p);
        append_removal(g, &mut s, &mut inradius, &incidents, &p);
        debug!(
            "\t-> greedy step: exclusion set len {}, inradius len {}",
            s.size(),
            inradius.len()
        );
        // TODO
        // 1. Find what should be the normal behavior: clearing or continue
        // updating the inradius set
        // 2. In the latter case, optimization to not re-allocate each time
        // since could be quite big with large k and radius
        if p.reset {
            inradius.clear();
        }
    }
    s
}

// append_removal is an adaptation of "SelectRemovalNodes" function in Algorithm 6
// of https://eprint.iacr.org/2018/944.pdf. Instead of returning the set of nodes
// to remove, it simply adds them to the given set.
fn append_removal(
    g: &Graph,
    set: &mut ExclusionSet,
    inradius: &mut NodeSet,
    incidents: &Vec<Pair>,
    params: &GreedyParams,
) {
    let radius = params.radius;
    let k = params.k;
    let iter = params.iter_topk;
    if radius == 0 {
        // take the node with the highest number of incident path
        set.insert(incidents.iter().max_by_key(|pair| pair.1).unwrap().0);
        return;
    }

    let mut count = 0;
    let mut excluded = 0;
    for node in incidents.iter() {
        if iter {
            // optim to add as much as possible nodes: goal is to add
            // as much as possible k nodes to S in each iteration.
            if count == k {
                trace!(
                    "\t\t- iteration on incidence included k nodes in radius
                -break!!"
                );
                break;
            }
        } else if count + excluded == k {
            // original behavior of the pseudo code from paper
            // we stop when we looked at the first top k entries
            break;
        }

        if inradius.contains(&node.0) {
            // difference with previous insertion is that we only include
            // nodes NOT in the radius set
            excluded += 1;
            trace!(
                "\t\t-- node {} included already in radius (len {})- skip to include in
            radius",
                node.0,
                inradius.len()
            );
            continue;
        }
        set.insert(node.0);
        update_radius_set(g, node.0, inradius, params);
        count += 1;
        trace!(
            "\t-> iteration {}: {} new node inserted ({} excluded) -> inradius {:?}",
            count,
            excluded,
            node.0,
            inradius.len(),
        );
    }
    // If we didn't find any good candidates, that means the inradius set
    // covers all the node already. In that case, we simply take the one
    // with the maximum incident paths.
    // We only take one node instead of k because this situation indicates
    // we covered a large portion of the graph, therefore, the nodes
    // added here don't add much value to S. For the sake of progressing in the
    // algorithm, we still add one ( so we can have a different inradius at the
    // next iteration).
    if count == 0 {
        debug!("\t\t-> added by default one node {}", incidents[0].0);
        set.insert(incidents[0].0);
        if !params.reset {
            update_radius_set(g, incidents[0].0, inradius, params);
        }
        count += 1;
    }

    debug!(
        "\t-> added {}/{} nodes in |S| = {:.2}, depth(G-S) =  {:.3}n",
        count,
        k,
        set.size(),
        (g.depth_exclude(&set) as f32) / (g.cap() as f32),
    );
}

fn compute_direct_nodes(g: &Graph, v: usize, rad: &NodeSet) -> Vec<usize> {
    return g.parents()[v]
        .par_iter()
        .chain(g.children()[v].par_iter())
        .filter(|&node| !rad.contains(node))
        .cloned()
        .collect::<Vec<usize>>();
}

/// update_radius_set fills the given inradius set with nodes that inside a radius
/// of the given node. Size of the radius is given radius. It corresponds to the
/// under-specified function "UpdateNodesInRadius" in algo. 6 of
/// https://eprint.iacr.org/2018/944.pdf
/// Function only exposed for benchmarking
/// NOTE: The `radius` shouldn't change across calls for the same `inradius` set,
/// that is, if we already have a node in `inradius` then we won't look for it
/// again because we assume we already found all its closest nodes within a
/// specified `radius` (if the `radius` increased across calls we would be missing
/// nodes that were farther away in comparison to earlier calls).
pub fn update_radius_set(g: &Graph, node: usize, inradius: &mut NodeSet, p: &GreedyParams) {
    let radius = p.radius;
    let mut closests: Vec<Node> = Vec::with_capacity(radius * 10);
    // FIXME: We should be able to better estimate the size of this scratch
    //  vector based on the `radius` and the average degree of the nodes.
    let mut tosearch: Vec<Node> = Vec::with_capacity(closests.capacity());
    // insert first the given node and then add the close nodes
    inradius.insert(node);
    tosearch.push(node);
    // do it recursively "radius" times
    for i in 0..radius {
        // grab all direct nodes of those already in radius "i"
        closests = if p.parallel {
            tosearch
                .par_iter()
                .fold(
                    || Vec::new(),
                    |mut acc, idx| {
                        compute_direct_nodes(g, *idx, inradius)
                            .into_iter()
                            .for_each(|x| acc.push(x));
                        acc
                    },
                )
                .reduce(
                    || Vec::new(),
                    |mut acc, set| {
                        // would be nice to use following but rayon doesn't
                        // accept to use &mut set
                        //acc.append(&mut set);
                        acc.extend(set);
                        acc
                    },
                )
        } else {
            closests.clear();
            // grab all direct nodes of those already in radius "i"
            for &v in tosearch.iter() {
                compute_direct_nodes(g, v, inradius)
                    .into_iter()
                    .for_each(|x| closests.push(x));
            }
            closests
        };
        tosearch.clear();
        for &mut node in &mut closests {
            if inradius.insert(node) {
                tosearch.push(node);
            }
        }
        trace!(
            "update radius {}: {} new nodes, total {}",
            i,
            tosearch.len(),
            inradius.len()
        );
    }
}

#[derive(Clone, Debug, Eq)]
pub struct Pair(usize, usize);

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.cmp(&other.1)
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
/// count_paths implements the CountPaths method in Algo. 5 for the greedy algorithm
/// It returns:
/// 1. the number of incident paths of the given length for each node.
///      Index is the the index of the node, value is the paths count.
/// 2. the top k nodes indexes that have the higest incident paths
///      The number of incident path is not given.
pub fn count_paths(g: &Graph, s: &ExclusionSet, p: &GreedyParams) -> Vec<Pair> {
    if p.use_degree {
        return count_paths_degree(g, s);
    }
    let length = p.length;
    // dimensions are [n][depth]
    // counting phase of all starting/ending paths of all length
    let (mut ending_paths, mut starting_paths): (Vec<Vec<u64>>, Vec<Vec<u64>>) = (0..g.size())
        .into_par_iter()
        .map(|node| {
            let mut ending_paths = vec![0 as u64; length + 1];
            let mut starting_paths = vec![0 as u64; length + 1];
            if !s.contains(node) {
                ending_paths[0] = 1;
                starting_paths[0] = 1;
            }
            (ending_paths, starting_paths)
        })
        .unzip();

    for d in 1..=length {
        g.for_each_edge(|e| {
            // checking each parents (vs only checking direct + 1parent in C#)
            // no ending path for node i if the parent is contained in S
            // since G - S doesn't have this parent
            if !s.contains(e.parent) {
                ending_paths[e.child][d] += ending_paths[e.parent][d - 1];

                // difference vs the pseudo code: like in C#, increase parent count
                // instead of iterating over children of node i
                starting_paths[e.parent][d] += starting_paths[e.child][d - 1];
            }
        });
    }

    // counting the top k node wo have the greatest number of incident paths
    // NOTE: difference with the C# that recomputes that vector separately.
    // Since topk is directly correlated to incidents[], we can compute both
    // at the same time and remove one O(n) iteration.
    let incident_of = |node: usize| -> Pair {
        Pair(
            node,
            (0..=length)
                .into_par_iter()
                .map(|d| (starting_paths[node][d] * ending_paths[node][length - d]) as usize)
                .sum(),
        )
    };

    // FIXME: this specific part doesn't improve much
    let mut incidents = if p.parallel {
        (0..g.size())
            .into_par_iter()
            .filter(|&n| !s.contains(n))
            .fold(
                || Vec::new(),
                |mut acc, n| {
                    acc.push(incident_of(n));
                    acc
                },
            )
            .reduce(
                || Vec::with_capacity(g.size()),
                |mut acc, p| {
                    acc.extend(p);
                    acc
                },
            )
    } else {
        (0..g.size()).into_iter().filter(|&n| !s.contains(n)).fold(
            Vec::with_capacity(g.size()),
            |mut acc, n| {
                acc.push(incident_of(n));
                acc
            },
        )
    };

    if p.parallel {
        // parallel sorting improves time
        incidents.par_sort_by_key(|pair| Reverse(pair.1));
    } else {
        incidents.sort_by_key(|pair| Reverse(pair.1));
    }
    incidents
}

fn count_paths_degree(g: &Graph, s: &ExclusionSet) -> Vec<Pair> {
    let mut v = Vec::with_capacity(g.size() - s.size());
    g.for_each_node(|&node| {
        if s.contains(node) {
            return;
        }
        let nc = g.children()[node]
            .iter()
            .filter(|&p| !s.contains(*p))
            .count();
        let np = g.parents()[node]
            .iter()
            .filter(|&p| !s.contains(*p))
            .count();
        v.push(Pair(node, nc + np));
    });
    v.sort_by_key(|a| Reverse(a.1));
    return v;
}
/// Implements the algorithm described in the Lemma 6.2 of the [AB16
/// paper](https://eprint.iacr.org/2016/115.pdf).
/// For a graph G with m edges, 2^k vertices, and \delta in-ground degree,
/// it returns a set S such that depth(G-S) <= 2^k-t
/// It iterates over t until depth(G-S) <= depth.
fn valiant_ab16(g: &Graph, target: usize) -> ExclusionSet {
    let mut s = ExclusionSet::new(g);
    // FIXME can we avoid that useless first copy ?
    let mut curr = g.remove(&s);
    loop {
        let partitions = valiant_partitions(&curr);
        // mi = # of edges at iteration i
        let mi = curr.count_edges();
        let depth = curr.depth();
        // depth at iteration i
        let di = depth.next_power_of_two();
        // power of exp. such that di <= 2^ki
        let ki = (di as f32).log2().ceil() as usize;
        let max_size = mi / ki;
        // take the minimum partition which has a size <= mi/ki
        let chosen: &EdgeSet = partitions
            .iter()
            .filter(|&partition| partition.len() > 0)
            .filter(|&partition| partition.len() <= max_size)
            .min_by_key(|&partition| partition.len())
            .unwrap();
        // TODO should this be even a condition to search for the partition ?
        // Paper claims it's always the case by absurd
        let new_depth = curr.depth_exclude_edges(chosen);
        assert!(new_depth <= (di >> 1));
        // G_i+1 = G_i - S_i  where S_i is set of origin nodes in chosen partition
        let mut si = ExclusionSet::new(&g);
        chosen.iter().for_each(|edge| si.insert(edge.parent));
        trace!(
        "m/k = {}/{} = {}, chosen = {:?}, new_depth {}, curr.depth() {}, curr.dpeth_exclude {}, new edges {}, si {:?}",
        mi,
        ki,
        max_size,
        chosen,
        new_depth,
        curr.depth(),
        curr.depth_exclude(&si),
        curr.count_edges(),
        si,
        );
        curr = curr.remove(&si);
        s.extend(&si);

        if curr.depth() <= target {
            trace!("\t -> breaking out, depth(G-S) = {}", g.depth_exclude(&s));
            break;
        }
    }
    return s;
}

fn valiant_reduce(g: &Graph, d: AttackAlgo) -> ExclusionSet {
    match d {
        // valiant_reduce returns a set S such that depth(G - S) < target.
        // It implements the algo 8 in the https://eprint.iacr.org/2018/944.pdf paper.
        AttackAlgo::ValiantDepth(depth) => {
            valiant_reduce_main(g, &|set: &ExclusionSet| g.depth_exclude(set) > depth)
        }
        AttackAlgo::ValiantSize(size) => {
            valiant_reduce_main(g, &|set: &ExclusionSet| set.size() < size)
        }
        AttackAlgo::ValiantAB16(depth) => valiant_ab16(g, depth),
        _ => panic!("that should not happen"),
    }
}

fn valiant_reduce_main(g: &Graph, f: &dyn Fn(&ExclusionSet) -> bool) -> ExclusionSet {
    let partitions = valiant_partitions(g);
    // TODO replace by a simple bitset or boolean vec
    let mut chosen: Vec<usize> = Vec::new();
    let mut s = ExclusionSet::new(g);
    // returns the smallest next partition unchosen
    // mut is required because it changes chosen which is mut
    let mut find_next = || -> &EdgeSet {
        match partitions
            .iter()
            .enumerate()
            // only take partitions with edges in it
            .filter(|&(_, values)| values.len() > 0)
            // only take the ones we didn't choose before
            .filter(|&(i, _)| !chosen.contains(&i))
            // take the smallest one
            .min_by_key(|&(_, values)| values.len())
        {
            Some((i, val)) => {
                chosen.push(i);
                val
            }
            None => panic!("no more partitions to use"),
        }
    };
    while f(&s) {
        let partition = find_next();
        // add the origin node for each edges in the chosen partition
        partition.iter().for_each(|edge| s.insert(edge.parent));
    }

    return s;
}

// valiant_partitions returns the sets E_i and S_i from the given graph
// according to the definition algorithm 8 from
// https://eprint.iacr.org/2018/944.pdf .
fn valiant_partitions(g: &Graph) -> Vec<EdgeSet> {
    let bs = utils::node_bitsize();
    let mut eis = Vec::with_capacity(bs);
    for _ in 0..bs {
        eis.push(EdgeSet::default());
    }

    g.for_each_edge(|edge| {
        let bit = utils::msbd(edge);
        debug_assert!(bit < bs);
        // edge j -> i differs at the nth bit
        eis[bit].insert(edge.clone());
    });

    eis
}

#[cfg(test)]
mod test {
    use super::super::graph;
    use super::*;
    use crate::graph::{DRGAlgo, Edge};
    use rand::Rng;

    use std::collections::HashSet;
    use std::iter::FromIterator;

    static TEST_SIZE: usize = 20;
    static TEST_MAX_PATH_LENGTH: usize = TEST_SIZE / 5;

    // graph 0->1->2->3->4->5->6->7
    // + 0->2 , 2->4, 4->6

    lazy_static! {
        static ref TEST_PARENTS: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1, 0],
            vec![2],
            vec![3, 2],
            vec![4],
            vec![5, 4],
            vec![6],
        ];
        static ref GREEDY_PARENTS: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1, 0],
            vec![2, 1],
            vec![3, 2, 0],
            vec![4],
        ];
    }

    #[test]
    fn test_greedy() {
        let mut graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        graph.children_project();
        let params = GreedyParams {
            k: 1,
            radius: 0,
            length: 2,
            ..GreedyParams::default()
        };
        let s = greedy_reduce(&mut graph, AttackAlgo::GreedyDepth(2, params));
        assert_eq!(s, ExclusionSet::from_nodes(&graph, vec![3, 4]));
        let params = GreedyParams {
            k: 1,
            radius: 1,
            length: 2,
            reset: true,
            ..GreedyParams::default()
        };
        let s = greedy_reduce(&mut graph, AttackAlgo::GreedyDepth(2, params));
        // + incidents [Pair(2, 7), Pair(4, 7), Pair(3, 6), Pair(0, 5), Pair(1, 5), Pair(5, 3)]
        //         -> iteration 1 : node 2 inserted -> inradius {0, 3, 1, 2, 4}
        //         -> added 1/6 nodes in |S| = 1, depth(G-S) = 4 = 0.667n
        // + incidents [Pair(3, 3), Pair(4, 3), Pair(0, 2), Pair(1, 2), Pair(5, 2), Pair(2, 0)]
        //         -> iteration 1 : node 3 inserted -> inradius {3, 1, 2, 4}
        //         -> added 1/6 nodes in |S| = 2, depth(G-S) = 2 = 0.333n
        //
        assert_eq!(s, ExclusionSet::from_nodes(&graph, vec![2, 3]));
        println!("\n\n\n ------\n\n\n");
        let params = GreedyParams {
            k: 1,
            radius: 1,
            reset: false,
            length: 2,
            ..GreedyParams::default()
        };
        let s = greedy_reduce(&mut graph, AttackAlgo::GreedyDepth(2, params));
        // iteration 1: incidents [Pair(2, 7), Pair(4, 7), Pair(3, 6), Pair(0, 5), Pair(1, 5), Pair(5, 3)]
        // -> iteration 1 : node 2 inserted -> inradius {0, 3, 1, 4, 2}
        // -> added 1/1 nodes in |S| = 1, depth(G-S) = 4 = 0.667n
        // Iteration 2: [Pair(3, 3), Pair(4, 3), Pair(0, 2), Pair(1, 2), Pair(5, 2), Pair(2, 0)]
        // -> added by default one node 3
        // -> added 1/1 nodes in |S| = 2, depth(G-S) = 2 = 0.333n
        //
        assert_eq!(s, ExclusionSet::from_nodes(&graph, vec![3, 2]));

        let random_bytes = rand::thread_rng().gen::<[u8; 32]>();
        let size = (2 as usize).pow(10);
        let depth = (0.25 * size as f32) as usize;
        let mut g3 = Graph::new(size, random_bytes, DRGAlgo::MetaBucket(3));
        let mut params = GreedyParams {
            k: 30,
            length: 8,
            radius: 2,
            iter_topk: true,
            reset: true,
            use_degree: false,
            parallel: false,
        };
        let set1 = greedy_reduce(&mut g3, AttackAlgo::GreedyDepth(depth, params.clone()));

        assert!(g3.depth_exclude(&set1) < depth);
        params.use_degree = true;
        let set2 = greedy_reduce(&mut g3, AttackAlgo::GreedyDepth(depth, params.clone()));
        assert!(g3.depth_exclude(&set2) < depth);
    }

    // FIXME: Update test description with new standardize order of `topk`
    // in `count_paths`.
    #[test]
    fn test_append_removal_node() {
        let mut graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        graph.children_project();
        let mut s = ExclusionSet::new(&graph);
        let mut params = GreedyParams {
            k: 3,
            length: 2,
            radius: 0,
            ..GreedyParams::default()
        };
        println!("graph: {:?}", graph);
        let incidents = count_paths(&graph, &s, &params);
        let mut inradius = NodeSet::default();
        append_removal(&graph, &mut s, &mut inradius, &incidents, &params);
        // incidents: [Pair(2, 7), Pair(4, 7), Pair(3, 6), Pair(0, 5), Pair(1, 5), Pair(5, 3)]
        //  only one value since radius == 0
        assert!(s.contains(4));

        params.radius = 1;
        let incidents = count_paths(&graph, &s, &params);
        println!("incidents: {:?}", incidents);
        append_removal(&graph, &mut s, &mut inradius, &incidents, &params);
        println!("s contains: {:?}", s);

        // [Pair(0, 3), Pair(1, 3), Pair(2, 3), Pair(3, 3), Pair(4, 0), Pair(5, 0)]
        // -> iteration 1 : node 0 inserted -> inradius {1, 2, 0, 4}
        //      - no other added since 0,1,2 makes k iteration
        //          "old behavior" only loops k times
        // NOTE:
        //  - 4 is already present thanks to last call
        assert_eq!(s, ExclusionSet::from_nodes(&graph, vec![4, 0]));
        // TODO probably more tests with larger graph
    }

    #[test]
    fn test_update_radius() {
        let mut graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        graph.children_project();
        let node = 2;
        let mut inradius = NodeSet::default();
        let mut p = GreedyParams {
            radius: 1,
            ..GreedyParams::default()
        };

        update_radius_set(&graph, node, &mut inradius, &p);
        assert_eq!(inradius, HashSet::from_iter(vec![0, 1, 2, 3, 4]));
        p.radius = 2;
        // Start another search with a bigger `radius`, clear previous
        // `inradius` to look for the nodes all over again.
        inradius.clear();
        update_radius_set(&graph, node, &mut inradius, &p);
        assert_eq!(inradius, HashSet::from_iter(vec![0, 1, 2, 3, 4, 5]));

        // start again with parallelism
        inradius.clear();
        p.parallel = true;
        p.radius = 1;
        update_radius_set(&graph, node, &mut inradius, &p);
        assert_eq!(inradius, HashSet::from_iter(vec![0, 1, 2, 3, 4]));
        inradius.clear();
        p.radius = 2;
        update_radius_set(&graph, node, &mut inradius, &p);
        assert_eq!(inradius, HashSet::from_iter(vec![0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_count_paths() {
        let graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        // test with empty set to remove
        let mut s = ExclusionSet::new(&graph);
        let p = GreedyParams {
            k: 3,
            length: 2,
            ..GreedyParams::default()
        };
        let incidents = count_paths(&graph, &s, &p);
        let mut exp = vec![
            Pair(0, 5),
            Pair(1, 5),
            Pair(2, 7),
            Pair(3, 6),
            Pair(4, 7),
            Pair(5, 3),
        ];
        exp.sort_by_key(|a| Reverse(a.1));

        assert_eq!(incidents, exp);
        s.insert(4);
        let incidents = count_paths(&graph, &s, &p);
        let mut exp = vec![Pair(0, 3), Pair(1, 3), Pair(2, 3), Pair(3, 3), Pair(5, 0)];
        exp.sort_by_key(|a| Reverse(a.1));
        assert_eq!(incidents, exp);
    }

    #[test]
    fn test_count_regular_connections() {
        let seed = [1; 32];
        // FIXME: Dummy seed, not used in `KConnector`, shouldn't be
        //  mandatory to provide it for graph construction (it should
        //  be part of the algorithm).

        for k in 1..TEST_SIZE {
            // When the `k` is to big stop. At least the center node
            // should see at both sides (`TEST_SIZE / 2`) paths of the
            // target length using any of the `k` connections, even
            // the longest one (of a distance of `k` nodes), so the
            // longest path overall should accommodate `k * length` nodes.
            if TEST_SIZE / 2 < k * TEST_MAX_PATH_LENGTH {
                break;
            }

            let g = Graph::new(TEST_SIZE, seed, DRGAlgo::KConnector(k));

            for length in 1..TEST_MAX_PATH_LENGTH {
                // The number of incident paths for the center node should be:
                // * Searching for a single length at one side of the node,
                //   `k^length`, since at any node that we arrive there be `k`
                //   new paths to discover.
                // * Splitting that length to both sides, we should still find
                //   that `k^partial_length` paths (which are joined multiplying
                //   them and reaching the `k^length` total), and we can divide
                //   the length in two in `length + 1` different ways (a length
                //   of zero on one side is valid, it means we look for the path
                //   in only one direction).
                let expected_count = k.pow(length as u32) * (length + 1);

                let p = GreedyParams {
                    k: 1,
                    length: length,
                    ..GreedyParams::default()
                };
                let incidents = count_paths(&g, &mut ExclusionSet::new(&g), &p);
                assert_eq!(
                    // find the value which corresponds to the middle node
                    incidents.iter().find(|&p| p.0 == g.size() / 2).unwrap().1,
                    expected_count
                );
                // FIXME: Extend the check for more nodes in the center of the graph.
            }
        }
    }

    #[test]
    fn test_valiant_reduce_depth() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let set = valiant_reduce(&graph, AttackAlgo::ValiantDepth(2));
        assert_eq!(set, ExclusionSet::from_nodes(&graph, vec![0, 2, 3, 4, 6]));
    }

    #[test]
    fn test_valiant_reduce_size() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let set = valiant_reduce(&graph, AttackAlgo::ValiantSize(3));
        assert_eq!(set, ExclusionSet::from_nodes(&graph, vec![0, 2, 3, 4, 6]));
    }

    #[test]
    fn test_valiant_ab16() {
        let parents = vec![
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
        ];

        let g = graph::tests::graph_from(parents);
        let target = 4;
        let set = valiant_reduce(&g, AttackAlgo::ValiantAB16(target));
        assert!(g.depth_exclude(&set) < target);
        // 3->4 differs at 3rd bit and they're the only one differing at that bit
        // so set s contains origin node 3
        assert_eq!(set, ExclusionSet::from_nodes(&g, vec![3]));

        let g = Graph::new(TEST_SIZE, graph::tests::TEST_SEED, DRGAlgo::MetaBucket(2));
        let target = TEST_SIZE / 4;
        let set = valiant_reduce(&g, AttackAlgo::ValiantAB16(target));
        assert!(g.depth_exclude(&set) <= target);
    }

    #[test]
    fn test_valiant_partitions() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let edges = valiant_partitions(&graph);
        assert_eq!(edges.len(), utils::node_bitsize());
        edges
            .into_iter()
            .enumerate()
            .for_each(|(i, edges)| match i {
                0 => {
                    assert_eq!(
                        edges,
                        HashSet::from_iter(vec![
                            Edge::new(0, 1),
                            Edge::new(2, 3),
                            Edge::new(4, 5),
                            Edge::new(6, 7)
                        ])
                    );
                }
                1 => {
                    assert_eq!(
                        edges,
                        HashSet::from_iter(vec![
                            Edge::new(0, 2),
                            Edge::new(1, 2),
                            Edge::new(4, 6),
                            Edge::new(5, 6)
                        ])
                    );
                }
                2 => {
                    assert_eq!(
                        edges,
                        HashSet::from_iter(vec![Edge::new(2, 4), Edge::new(3, 4)])
                    );
                }
                _ => {}
            });
    }

    #[test]
    fn greedy_k_ratio() {
        let size = 20; // n = 2^20
        let k = GreedyParams::k_ratio(size as usize);
        assert_eq!(k, 800);
    }
}
