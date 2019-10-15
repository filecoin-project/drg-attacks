use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::time::Instant;

use log::{debug, trace};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use serde_json::Result;

use crate::graph::{DRGAlgo, Edge, EdgeSet, ExclusionSet, Graph, GraphSpec, Node, NodeSet};
use crate::utils;

// FIXME: This name is no longer representative, we no longer attack using
//  depth as a target, we also have a size target now. This should be renamed
//  to something more generic like `AttackType`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DepthReduceSet {
    /// depth of the resulting G-S graph desired
    ValiantDepth(usize),
    /// size of the resulting S desired
    ValiantSize(usize),
    /// AB16 Lemma 6.2 variant of the Valiant Lemma's based attack.
    /// Parameter  is size of the resulting G-S graph desired
    ValiantAB16(usize),
    /// depth of the resulting G-S graph and some specific parameters
    GreedyDepth(usize, GreedyParams),
    /// Variation of Greedy attack that has as `target` the resulting size of set S.
    GreedySize(usize, GreedyParams),
}

pub fn depth_reduce(g: &mut Graph, drs: DepthReduceSet) -> ExclusionSet {
    match drs {
        DepthReduceSet::ValiantDepth(_) => valiant_reduce(g, drs),
        DepthReduceSet::ValiantSize(_) => valiant_reduce(g, drs),
        DepthReduceSet::ValiantAB16(_) => valiant_reduce(g, drs),
        DepthReduceSet::GreedyDepth(_, _) => greedy_reduce(g, drs),
        DepthReduceSet::GreedySize(_, _) => greedy_reduce(g, drs),
    }
}

/// Target of an attack, either the depth should be smaller than `Depth` or
/// the exclusion set `S` should have a size bigger than `Size` (actually
/// we want to hit a target as close as possible as those thresholds). All
/// targets are specified in relation to the size of the graph `G` (not to
/// be confused with the target size of set `S`).
// FIXME: Overlapping with `DepthReduceSet` internal values, this should
//  replace that.
#[derive(Debug)]
pub enum AttackTarget {
    Depth(f64),
    Size(f64),
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
    pub target: AttackTarget,
    pub range: TargetRange,
    pub attack: DepthReduceSet,
}

impl AttackProfile {
    /// Build a default profile from an attack type that has only one run
    /// in a range of a single value (to make it compatible with previous
    /// uses of `attack`).
    // FIXME: We shouldn't need the `graph_size` (or the graph for that
    // matter), but this is accommodating previous uses of `attack` (which
    // should be refactored entirely and this method removed or reworked).
    pub fn from_attack(attack: DepthReduceSet, graph_size: usize) -> Self {
        let graph_size = graph_size as f64;
        let target = match attack {
            DepthReduceSet::ValiantDepth(depth) => AttackTarget::Depth(depth as f64 / graph_size),
            DepthReduceSet::ValiantSize(size) => AttackTarget::Size(size as f64 / graph_size),
            DepthReduceSet::ValiantAB16(size) => AttackTarget::Size(size as f64 / graph_size),
            DepthReduceSet::GreedyDepth(depth, _) => AttackTarget::Depth(depth as f64 / graph_size),
            DepthReduceSet::GreedySize(size, _) => AttackTarget::Size(size as f64 / graph_size),
        };
        // FIXME: This code should absorb the `depth_reduce` and derived
        // functions logic. The target discrimination depth/size should
        // b independent of the attack type (valiant/greedy).

        let range = {
            let single_value = match target {
                AttackTarget::Depth(depth) => depth,
                AttackTarget::Size(size) => size,
            };
            // FIXME: Too verbose, there probably is a more concise way to do this.
            TargetRange {
                start: single_value,
                end: single_value,
                interval: 0.0,
            }
        };

        AttackProfile {
            runs: 1,
            target,
            range,
            attack,
        }
    }
}

/// Results of an attack expressed in relation to the graph size.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct SingleAttackResult {
    depth: f64,
    exclusion_size: f64,
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
    target: f64,
    mean_depth: f64,
    mean_size: f64,
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
#[derive(Serialize, Deserialize)]
pub struct AttackResults {
    results: Vec<AveragedAttackResult>,
    attack: DepthReduceSet,
}

impl std::fmt::Display for SingleAttackResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "\t-> |S| = {:.4}\n\t-> depth(G-S) = {:.4}",
            self.exclusion_size, self.depth,
        )
    }
}

pub fn attack(g: &mut Graph, attack: DepthReduceSet) -> SingleAttackResult {
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
        let mut g = Graph::new_from_rng(spec, &mut rng);

        for (t, target) in targets.iter().enumerate() {
            let absolute_target = (target * spec.size as f64) as usize;
            let attack_type = match profile.attack.clone() {
                DepthReduceSet::ValiantDepth(_) => DepthReduceSet::ValiantDepth(absolute_target),
                DepthReduceSet::ValiantSize(_) => DepthReduceSet::ValiantSize(absolute_target),
                DepthReduceSet::ValiantAB16(_) => DepthReduceSet::ValiantAB16(absolute_target),
                DepthReduceSet::GreedyDepth(_, p) => {
                    DepthReduceSet::GreedyDepth(absolute_target, p)
                }
                DepthReduceSet::GreedySize(_, p) => DepthReduceSet::GreedySize(absolute_target, p),
            };
            // FIXME: Same as before, the target should be decoupled from the type of attack.

            println!(
                "Attack (run {}) target ({:?} = {}), with {:?}",
                run, profile.target, target, attack_type
            );
            results[t][run] = attack(&mut g, attack_type.clone());
        }
    }

    AttackResults {
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
    pub fn k_ratio(size: usize) -> usize {
        assert!(size >= 20);
        (2 as usize).pow((size as u32 - 18) / 2) * 400
    }
}

// greedy_reduce implements the Algorithm 5 of https://eprint.iacr.org/2018/944.pdf
fn greedy_reduce(g: &mut Graph, d: DepthReduceSet) -> ExclusionSet {
    match d {
        DepthReduceSet::GreedyDepth(depth, p) => {
            greedy_reduce_main(g, p, &|set: &ExclusionSet, g: &mut Graph| {
                g.depth_exclude(set) > depth
            })
        }
        DepthReduceSet::GreedySize(size, p) => {
            // FIXME: To hit exactly the `target_size` we should consider the number of nodes
            //  removed in each iteration (`GreedyParams::k`), but since that number is small
            //  compared to normal target sizes it is an acceptable bias for now. We only
            //  correct `k` if it's bigger than 1/100th the target size.
            let mut p = p.clone();
            p.k = std::cmp::min(p.k, (size as f32 * 0.01).ceil() as usize);

            greedy_reduce_main(g, p, &|set: &ExclusionSet, g: &mut Graph| set.size() < size)
        }
        _ => panic!("invalid DepthReduceSet option"),
    }
}

fn greedy_reduce_main(
    g: &mut Graph,
    p: GreedyParams,
    f: &Fn(&ExclusionSet, &mut Graph) -> bool,
) -> ExclusionSet {
    let mut s = ExclusionSet::new(g);
    g.children_project();
    let mut inradius: NodeSet = NodeSet::default();
    while f(&s, g) {
        // TODO use p.length when more confidence in the trick
        let incidents = count_paths(g, &s, &p);
        append_removal(g, &mut s, &mut inradius, &incidents, &p);

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
            continue;
        }
        set.insert(node.0);
        update_radius_set(g, node.0, inradius, radius);
        count += 1;
        debug!(
            "\t-> iteration {} : node {} inserted -> inradius {:?}",
            count + excluded,
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
        debug!("\t-> added by default one node {}", incidents[0].0);
        set.insert(incidents[0].0);
        update_radius_set(g, incidents[0].0, inradius, radius);
        count += 1;
    }

    let d = g.depth_exclude(&set);
    debug!(
        "\t-> added {}/{} nodes in |S| = {}, depth(G-S) = {} = {:.3}n",
        count,
        k,
        set.size(),
        d,
        (d as f32) / (g.cap() as f32),
    );
}

// update_radius_set fills the given inradius set with nodes that inside a radius
// of the given node. Size of the radius is given radius. It corresponds to the
// under-specified function "UpdateNodesInRadius" in algo. 6 of
// https://eprint.iacr.org/2018/944.pdf
// NOTE: The `radius` shouldn't change across calls for the same `inradius` set,
// that is, if we already have a node in `inradius` then we won't look for it
// again because we assume we already found all its closest nodes within a
// specified `radius` (if the `radius` increased across calls we would be missing
// nodes that were farther away in comparison to earlier calls).
fn update_radius_set(g: &Graph, node: usize, inradius: &mut NodeSet, radius: usize) {
    let mut closests: Vec<Node> = Vec::with_capacity(radius * 10);
    // FIXME: We should be able to better estimate the size of this scratch
    //  vector based on the `radius` and the average degree of the nodes.
    let mut tosearch: Vec<Node> = Vec::with_capacity(closests.capacity());

    let add_direct_nodes = |v: usize, closests: &mut Vec<Node>, _: &NodeSet| {
        // add all direct parent
        g.parents()[v]
            .iter()
            // no need to continue searching with that parent since it's
            // already in the radius, i.e. it already has been searched
            // FIXME see if it works and resolves any potential loops
            //.filter(|&parent| !rad.contains(parent))
            .for_each(|&parent| {
                closests.push(parent);
            });

        // add all direct children
        g.children()[v]
            .iter()
            // no need to continue searching with that parent since it's
            // already in the radius, i.e. it already has been searched
            //.filter(|&child| !rad.contains(child))
            .for_each(|&child| {
                closests.push(child);
            });
        trace!(
            "\t add_direct node {}: at most {} parents and {} children",
            v,
            g.parents()[v].len(),
            g.children()[v].len()
        );
    };
    // insert first the given node and then add the close nodes
    inradius.insert(node);
    tosearch.push(node);
    // do it recursively "radius" times
    for i in 0..radius {
        closests.clear();
        // grab all direct nodes of those already in radius "i"
        for &v in tosearch.iter() {
            add_direct_nodes(v, &mut closests, inradius);
        }
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
struct Pair(usize, usize);

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
// count_paths implements the CountPaths method in Algo. 5 for the greedy algorithm
// It returns:
// 1. the number of incident paths of the given length for each node.
//      Index is the the index of the node, value is the paths count.
// 2. the top k nodes indexes that have the higest incident paths
//      The number of incident path is not given.
fn count_paths(g: &Graph, s: &ExclusionSet, p: &GreedyParams) -> Vec<Pair> {
    if p.use_degree {
        return count_paths_degree(g, s, p);
    }
    let length = p.length;
    // dimensions are [n][depth]
    let mut ending_paths = vec![vec![0 as u64; length + 1]; g.cap()];
    let mut starting_paths = vec![vec![0 as u64; length + 1]; g.cap()];
    // counting phase of all starting/ending paths of all length

    for node in 0..g.size() {
        if !s.contains(node) {
            // initializes the tables with 1 for nodes present in G - S
            ending_paths[node][0] = 1;
            starting_paths[node][0] = 1;
        }
    }

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

    // counting how many incident paths of length d there is for each node
    let mut incidents = Vec::with_capacity(g.size());
    // counting the top k node wo have the greatest number of incident paths
    // NOTE: difference with the C# that recomputes that vector separately.
    // Since topk is directly correlated to incidents[], we can compute both
    // at the same time and remove one O(n) iteration.
    g.for_each_node(|&node| {
        if s.contains(node) {
            return;
        }
        incidents.push(Pair(
            node,
            (0..=length)
                .map(|d| (starting_paths[node][d] * ending_paths[node][length - d]) as usize)
                .sum(),
        ));
    });

    incidents.sort_by_key(|pair| Reverse(pair.1));
    incidents
}

fn count_paths_degree(g: &Graph, s: &ExclusionSet, p: &GreedyParams) -> Vec<Pair> {
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

fn valiant_reduce(g: &Graph, d: DepthReduceSet) -> ExclusionSet {
    match d {
        // valiant_reduce returns a set S such that depth(G - S) < target.
        // It implements the algo 8 in the https://eprint.iacr.org/2018/944.pdf paper.
        DepthReduceSet::ValiantDepth(depth) => {
            valiant_reduce_main(g, &|set: &ExclusionSet| g.depth_exclude(set) > depth)
        }
        DepthReduceSet::ValiantSize(size) => {
            valiant_reduce_main(g, &|set: &ExclusionSet| set.size() < size)
        }
        DepthReduceSet::ValiantAB16(depth) => valiant_ab16(g, depth),
        _ => panic!("that should not happen"),
    }
}

fn valiant_reduce_main(g: &Graph, f: &Fn(&ExclusionSet) -> bool) -> ExclusionSet {
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
    use rand::Rng;
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
        let s = greedy_reduce(&mut graph, DepthReduceSet::GreedyDepth(2, params));
        assert_eq!(s, ExclusionSet::from_nodes(&graph, vec![3, 4]));
        let params = GreedyParams {
            k: 1,
            radius: 1,
            length: 2,
            reset: true,
            ..GreedyParams::default()
        };
        let s = greedy_reduce(&mut graph, DepthReduceSet::GreedyDepth(2, params));
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
        let s = greedy_reduce(&mut graph, DepthReduceSet::GreedyDepth(2, params));
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
        };
        let set1 = greedy_reduce(&mut g3, DepthReduceSet::GreedyDepth(depth, params.clone()));

        assert!(g3.depth_exclude(&set1) < depth);
        params.use_degree = true;
        let set2 = greedy_reduce(&mut g3, DepthReduceSet::GreedyDepth(depth, params.clone()));
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

        update_radius_set(&graph, node, &mut inradius, 1);
        assert_eq!(inradius, HashSet::from_iter(vec![0, 1, 2, 3, 4]));

        // Start another search with a bigger `radius`, clear previous
        // `inradius` to look for the nodes all over again.
        inradius.clear();
        update_radius_set(&graph, node, &mut inradius, 2);
        assert_eq!(inradius, HashSet::from_iter(vec![0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_count_paths() {
        let graph = graph::tests::graph_from(GREEDY_PARENTS.to_vec());
        let target_length = 2;
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
        let set = valiant_reduce(&graph, DepthReduceSet::ValiantDepth(2));
        assert_eq!(set, ExclusionSet::from_nodes(&graph, vec![0, 2, 3, 4, 6]));
    }

    #[test]
    fn test_valiant_reduce_size() {
        let graph = graph::tests::graph_from(TEST_PARENTS.to_vec());
        let set = valiant_reduce(&graph, DepthReduceSet::ValiantSize(3));
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
        let set = valiant_reduce(&g, DepthReduceSet::ValiantAB16(target));
        assert!(g.depth_exclude(&set) < target);
        // 3->4 differs at 3rd bit and they're the only one differing at that bit
        // so set s contains origin node 3
        assert_eq!(set, ExclusionSet::from_nodes(&g, vec![3]));

        let g = Graph::new(TEST_SIZE, graph::tests::TEST_SEED, DRGAlgo::MetaBucket(2));
        let target = TEST_SIZE / 4;
        let set = valiant_reduce(&g, DepthReduceSet::ValiantAB16(target));
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
