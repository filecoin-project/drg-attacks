use fnv::FnvHasher;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::error;
use std::fmt;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hash};

/// Data that completely specifies the `Graph` to be created. Many runs
/// from the save stored data should produce the same `Graph` always
/// (that is, the same parents/edges).
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct GraphSpec {
    pub size: usize,
    // FIXME: Not always needed, should be behind an `Option`.
    // FIXME: We should create the graphs from a RNG *always* instead
    //  of directly from the seed (otherwise repeatedly created graphs
    //  for the same attack profile will be all the same). This should
    //  be enforced here.
    pub seed: [u8; 32],
    pub algo: DRGAlgo,
}

// Graph holds the parameters and the edges of the graph. This is a special type
// of graph that has a *proper* labelling: for each edge (i,j), we have i < j.
#[derive(Debug, Serialize, Deserialize)]
pub struct Graph {
    // parents holds all parents relationships of all nodes.
    // If j = parents[i][u] (for any u), then there is an edge (j -> i),
    // i.e. j is the parent of i.
    // The capacity of the graph is the size of the vector - Some nodes may be
    // absent when counted, i.e. node i may not have any parent and may not be
    // the parent of any other node. In that case, it is not included in the graph G
    parents: Vec<Vec<Node>>,
    // FIXME: Use slices, after construction this doesn't change.
    spec: GraphSpec,

    // children holds all the children relationships of all nodes.
    // If j = children[i][u] for any u, then there is an edge (i -> j).
    // NOTE: it is NOT computed by default, only when calling children_project()
    children: Vec<Vec<usize>>,
}

pub type Node = usize;

/// An edge represented as a parent-child relation (an expansion of the short
/// `(u,v)` notation used in the paper).
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Edge {
    pub parent: Node,
    pub child: Node,
}

impl Edge {
    pub fn new(parent: Node, child: Node) -> Edge {
        debug_assert!(
            parent < child,
            format!("the parent {} is not smaller than child {}", parent, child)
        );
        Edge { parent, child }
    }

    pub fn interval(&self) -> usize {
        self.child - self.parent - 1
    }

    pub fn active(&self, s: &ExclusionSet) -> bool {
        !s.contains(self.parent) && !s.contains(self.child)
    }

    pub fn inside_interval(&self, node: Node) -> bool {
        node > self.parent && node < self.child
    }

    pub fn closest_to(&self, node: Node) -> Node {
        assert!(self.inside_interval(node));
        if node - self.parent > self.child - node {
            self.child
        } else {
            self.parent
        }
    }

    pub fn overlaps(&self, other: &Edge) -> bool {
        // Easier to think of non-overlapping cases:
        !(self.child < other.parent || self.parent > other.child)
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.parent, self.child)
    }
}

/// Faster hasher than the default implementation to speed up hash set use
/// according to:
/// * https://blog.rust-lang.org/2016/03/02/Rust-1.7.html#library-stabilizations
// FIXME: Research more up-to-date solutions (or drop the `HashSet` entirely).
pub type FastHashSet<T> = HashSet<T, BuildHasherDefault<FnvHasher>>;
pub type NodeSet = FastHashSet<Node>;
pub type EdgeSet = FastHashSet<Edge>;

// DRGAlgo represents which algorithm can be used to create the edges so a Graph is
// a Depth Robust Graph
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum DRGAlgo {
    // BucketSample is the regular bucket sampling algorithm with degree 2
    BucketSample,
    // MetaBucket is the meta-graph construction with a specified degree
    MetaBucket(usize),
    /// Each node is connected to the `k` closest neighbors (including immediate
    /// predecessor). The objective is to have a guaranteed in/out-degree of `k`
    /// for each node (except for the nodes at the edges), the fact that we choose
    /// the *closest* ones is just to guarantee that with a regular topology. This
    /// algorithm is mostly for testing purposes.
    KConnector(usize),
    /// Uniform graph with all nodes with the same `m` edges, each with an interval
    /// `ner` times bigger than the last.
    UniformGraph {
        m: usize,
        ner: usize,
    },
}

/// Range used for a uniform distribution sample in `Rng::gen_range`: `[low, high)`.
#[derive(Debug, PartialEq)]
pub struct UniformSampleRange {
    pub low: usize,
    pub high: usize,
}

/// Ranges used for random sample of a generated parent. The `bucket` range is
/// deterministic, it only depends on the meta node, while the `node` range is
/// not, it depends on the bucket samples in the previous range.
#[derive(Debug)]
pub struct DRSampleRanges {
    bucket: UniformSampleRange,
    node: UniformSampleRange,
}

/// Abstraction over `Rng::gen_range`, which is the only thing we use from
/// that interface, in order to replace it with a fake RNG for testing purposes.
/// (We explicitly avoid reusing the `gen_range` name just to avoid multiple
/// candidates issues.)
trait UniformSampling {
    fn sample_range(&mut self, range: &UniformSampleRange) -> usize;
}
impl<R> UniformSampling for R
where
    R: Rng,
{
    fn sample_range(&mut self, range: &UniformSampleRange) -> usize {
        self.gen_range(range.low, range.high)
    }
}

/// Exclusion set `S` of nodes that are removed from `G`. Encapsulated in this
/// interface to evaluate optimizations to its implementation (e.g., set vs vec).
#[derive(Debug, Clone, PartialEq)]
pub struct ExclusionSet {
    v: Vec<bool>,
    size: usize,
}

impl ExclusionSet {
    /// Create new set `S` for a specified `size`.
    pub fn new_with_size(size: usize) -> Self {
        ExclusionSet {
            v: vec![false; size],
            size: 0,
        }
    }

    /// Create new set `S` for a specified `G`.
    pub fn new(graph: &Graph) -> Self {
        Self::new_with_size(graph.size())
    }

    pub fn from_nodes(graph: &Graph, nodes: Vec<Node>) -> Self {
        let mut es = Self::new(graph);
        for node in nodes {
            es.insert(node);
        }
        es
    }

    pub fn contains(&self, node: Node) -> bool {
        self.v[node]
    }

    // FIXME: Rename to remove_node (and same with `remove` which
    // is confusing, it should be restore node).
    pub fn insert(&mut self, node: Node) -> bool {
        if !self.contains(node) {
            self.v[node] = true;
            self.size += 1;
            true
        } else {
            false
        }
    }

    pub fn remove(&mut self, node: Node) -> bool {
        if self.contains(node) {
            self.v[node] = false;
            self.size -= 1;
            true
        } else {
            false
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn extend(&mut self, es: &ExclusionSet) {
        assert!(
            self.v.len() == es.v.len(),
            "exclusion set len mismatch when extending"
        );
        for node in 0..self.v.len() {
            if !self.contains(node) && es.contains(node) {
                self.v[node] = true;
                self.size += 1;
            }
        }
    }
}

impl Graph {
    // new returns a new graph instance from the given parameters.
    // The graph's edges are not generated yet, call fill_drg to compute the edges.
    pub fn new(size: usize, seed: [u8; 32], algo: DRGAlgo) -> Graph {
        // FIXME: To avoid changing too much code at the moment the `GraphSpec`
        //  is built here, but ideally the consumer should already provide it.
        let spec = GraphSpec { seed, size, algo };
        let mut rng = ChaChaRng::from_seed(spec.seed);

        Self::new_from_rng(spec, &mut rng)
    }

    // FIXME: The RNG is not always necessary so this function is misleading.
    pub fn new_from_rng(spec: GraphSpec, rng: &mut ChaChaRng) -> Graph {
        let mut g = Graph {
            spec,
            parents: Vec::with_capacity(spec.size),
            children: vec![],
        };
        match g.algo() {
            DRGAlgo::BucketSample => g.bucket_sample(rng),
            DRGAlgo::MetaBucket(degree) => g.meta_bucket(degree, rng),
            DRGAlgo::KConnector(k) => g.connect_neighbors(k),
            DRGAlgo::UniformGraph {
                m,
                ner,
            } => g.uniform(m, ner),
        }
        g
    }

    /// load_or_create tries to read the json description of the graph specified
    /// by the first argument. If it fails, it creates the graph by passing
    /// the rest of the argumetn to Graph::new, and saves the graph at the
    /// specified location.
    /// FIXME: why is it still taking so much time..
    pub fn load_or_create(fname: &str, size: usize, seed: [u8; 32], algo: DRGAlgo) -> Graph {
        if let Ok(graph) = Graph::load(fname) {
            println!("graph loaded from {}", fname);
            if graph.cap() == size {
                return graph;
            }
        }
        let g = Graph::new(size, seed, algo);
        g.save(fname);
        println!("graph created and saved at {}", fname);
        g
    }

    fn load(fname: &str) -> Result<Graph, Box<dyn error::Error>> {
        // Open the file in read-only mode with buffer.
        let file = File::open(fname)?;

        let g = serde_json::from_reader(file)?;
        Ok(g)
    }

    fn save(&self, fname: &str) {
        let file =
            File::create(fname).expect(format!("unable to save graph to {}", fname).as_str());

        serde_json::to_writer(file, self)
            .expect(format!("unable to save graph to {}", fname).as_str());
    }

    /// Number of nodes in the graph.
    // FIXME: Standardize size usage, don't access length or capacity of inner structures.
    pub fn size(&self) -> usize {
        self.spec.size
    }

    // depth_exclude returns the depth of the graph when excluding the given
    // set of nodes
    pub fn depth_exclude(&self, set: &ExclusionSet) -> usize {
        self.parents
            .iter()
            .enumerate()
            .fold(Vec::new(), |mut acc, (i, parents)| {
                if set.contains(i) {
                    // an excluded node has length 0
                    acc.push(0);
                    return acc;
                }
                match parents
                    .iter()
                    // dont take parent's length if contained in set
                    .filter(|&p| !set.contains(*p))
                    .map(|&p| acc[p] + 1)
                    .max()
                {
                    // need the match because there might not be any values
                    Some(depth) => acc.push(depth),
                    None => acc.push(1),
                }
                acc
            })
            .into_iter()
            .max()
            .unwrap()
    }

    // Depth of zero means excluded node which we don't connect, depth of one
    // is a single node, depth of two is an edge.
    // FIXME: Review the use and usefulness of this function.
    pub fn depth_exclude_with_edges(&self, set: &ExclusionSet) -> Vec<Edge> {
        let mut partial_depth = vec![0; self.size()];
        for child in 0..self.size() {
            if set.contains(child) {
                continue;
            }

            let mut parent_depths = vec![0; self.degree()];
            for (i, &parent) in self.parents()[child].iter().enumerate() {
                if set.contains(parent) {
                    continue;
                }
                parent_depths[i] = partial_depth[parent];
            }

            partial_depth[child] = match parent_depths.iter().max() {
                Some(max) => *max + 1,
                None => 1, // No parents, single node, depth of 1.
            };
        }

        let depth = *partial_depth.iter().max().unwrap();

        if depth <= 1 {
            return Vec::new();
        }

        // Recover the edges, if we have multiple maximums always choose the
        // first one (for simplicity).
        // Go back to find the end node of the largest path.
        let mut path_end = self.size() - 1;
        loop {
            if partial_depth[path_end] == depth {
                break;
            }
            path_end -= 1;
            // We should never be below zero (Rust would take care to panic in that scenario).
        }
        assert!(path_end < self.size());

        let mut path = Vec::with_capacity(depth);
        let mut child = path_end;
        let mut path_depth = depth;
        loop {
            if path_depth <= 1 {
                break;
            }

            path_depth -= 1;
            let mut parent = self.size();
            for &p in self.parents()[child].iter() {
                if partial_depth[p] == path_depth {
                    parent = p;
                    break;
                }
            }
            assert!(parent < self.size());

            path.push(Edge { parent, child });
            child = parent;
        }

        assert_eq!(depth - 1, path.len());

        path.reverse();
        path
    }

    // depth returns the longest depth found in the graph
    pub fn depth(&self) -> usize {
        self.parents
            .iter()
            .fold(Vec::new(), |mut acc, parents| {
                // take the depth of each parents + 1 then take the max of it
                match parents.iter().map(|&p| acc[p] + 1).max() {
                    Some(depth) => acc.push(depth),
                    None => acc.push(1),
                };
                acc
            })
            .into_iter()
            .max()
            .unwrap()
    }

    /// Returns the depth of the graph when removing the given edges
    // TODO: Is it possible to use traits to implement the equivalent of
    // function overloading as to get only one "depth_exclude" that works
    // for both types ?
    pub fn depth_exclude_edges(&self, edges: &EdgeSet) -> usize {
        // transform set of edges into list of parent relationship
        let edges_map = edges.iter().fold(HashMap::new(), |mut acc, edge| {
            (*acc.entry(edge.child).or_insert(Vec::new())).push(edge.parent);
            acc
        });

        self.parents
            .iter()
            .enumerate()
            .fold(Vec::new(), |mut acc, (child, parents)| {
                match parents
                    .iter()
                    // filter all parents which are in the list of edges to remove
                    .filter(|&parent| match edges_map.get(&child) {
                        None => true,
                        Some(pparents) => !pparents.contains(parent),
                    })
                    .map(|&parent| acc[parent] + 1)
                    .max()
                {
                    Some(depth) => acc.push(depth),
                    None => acc.push(0),
                }
                acc
            })
            .into_iter()
            .max()
            .unwrap()
    }

    // remove returns a new graph with the specified nodes removed
    // TODO slow path checking in O(n) - consider using bitset for nodes
    pub fn remove(&self, nodes: &ExclusionSet) -> Graph {
        let mut out = Vec::with_capacity(self.parents.len());
        for i in 0..self.parents.len() {
            let parents = self.parents.get(i).unwrap();
            let new_parents = if nodes.contains(i) {
                // no parent for a deleted node
                Vec::new()
            } else {
                // only take parents which are not in the list of nodes
                parents
                    .into_iter()
                    .filter(|&parent| !nodes.contains(*parent))
                    .map(|&p| p)
                    .collect::<Vec<usize>>()
            };
            out.push(new_parents);
        }

        Graph {
            spec: GraphSpec {
                size: (&out).len(),
                ..self.spec
            },
            // FIXME: We should think if we actually need to create new graphs
            //  out of old ones (just to count the depth of a reduced set).
            //  This violates the contract of `GraphSpec`: this new graph
            //  wouldn't be able to be recreated from it.
            parents: out,
            children: vec![],
        }
    }

    fn algo(&self) -> DRGAlgo {
        self.spec.algo
    }

    // Implementation of the first algorithm BucketSample on page 22 of the
    // porep paper : https://web.stanford.edu/~bfisch/porep_short.pdf
    // It produces a degree-2 graph which is asymptotically depth-robust.
    fn bucket_sample(&mut self, rng: &mut ChaChaRng) {
        for node in 0..self.parents.capacity() {
            let mut parents = Vec::new();
            match node {
                // no parents for the first node
                0 => {}
                // second node only has the first node as parent
                1 => {
                    parents.push(0);
                }
                _ => {
                    // push the direct parent of i, i.e. (i-1 -> i)
                    parents.push(node - 1);

                    // choose a bucket index
                    let max_bucket = (node as f32).log2().ceil() as usize;
                    let i: usize = rng.gen_range(1, max_bucket + 1);
                    // get a node from that bucket, i.e. from [2^i-1, 2^i[
                    // exclusif because otherwise a parent can be the same
                    // as its child
                    let max = std::cmp::min(node, 1 << i);
                    let min = std::cmp::max(2, max >> 1);
                    let random_parent = node - rng.gen_range(min, max + 1);
                    assert!(random_parent < node);
                    parents.push(random_parent);
                }
            }

            remove_duplicate(&mut parents);
            self.parents.push(parents);
        }
    }

    // Implementation of the meta-graph construction algorithm described in page 22
    // of the porep paper https://web.stanford.edu/~bfisch/porep_short.pdf
    // It produces a degree-d graph on average.
    fn meta_bucket(&mut self, degree: usize, rng: &mut ChaChaRng) {
        let m = degree - 1;
        for node in 0..self.parents.capacity() {
            let mut parents = Vec::with_capacity(degree);
            match node {
                // no parents for the first node
                0 => {}
                // second node only has the first node as parent
                1 => {
                    parents.push(0);
                }
                _ => {
                    // push the direct parent of i, i.e. (i-1 -> i)
                    parents.push(node - 1);

                    // similar to bucket_sample but we select m parents instead
                    // of just one
                    parents.extend((0..m).map(|_| Self::sample_parent_node(node, m, rng).0));
                }
            }
            // filtering duplicate parents
            remove_duplicate(&mut parents);
            self.parents.push(parents);
        }
    }

    /// Core of the meta-graph construction (`meta_bucket`) isolated for audit and
    /// test purposes: samples *one* parent of a node. Parameters:
    /// * `node`: Index of the original node we're assigning a parent to.
    /// * `m`: Target base degree for each node *without* counting direct predecessor.
    /// * `rng`: RNG used *twice*, for bucket selection and posterior node selection
    ///           (within that bucket).
    /// Returns:
    /// * Sampled parent.
    /// * Ranges used in the uniform sample to arrive to that parent (for testing
    ///    purposes only, can be safely ignore elsewhere).
    // FIXME: Revisit the name.
    fn sample_parent_node<R>(node: usize, m: usize, rng: &mut R) -> (usize, DRSampleRanges)
    where
        R: UniformSampling,
    {
        // meta_idx represents a meta node in the meta graph
        // each node is represented m times, so we always take the
        // first node index to not fall on the same final index
        // graphically, it looks like
        // [ [node 0, ..., node 0, node 1 ... node 1, etc]
        // with each "bucket" of node having length
        let meta_idx = node * m;
        // ceil instead of floor() + 1
        let max_bucket = (meta_idx as f32).log2().ceil() as usize;
        // choose bucket index {1 ... ceil(log2(idx))}
        let bucket_range = UniformSampleRange {
            low: 1,
            high: max_bucket + 1,
        };
        let i: usize = rng.sample_range(&bucket_range);
        // choose parent in range [min(2, 2^(i-1)), max(meta,2^i)[

        // min to avoid choosing a node which is higher than
        // the meta_idx - can happen since we can choose one
        // in the same bucket!
        let max = std::cmp::min(meta_idx, 1 << i);
        let min = std::cmp::max(2, max >> 1);
        assert!(max <= meta_idx);
        let node_range = UniformSampleRange {
            low: min,
            high: max + 1,
        };
        let meta_parent = meta_idx - rng.sample_range(&node_range);
        let real_parent = meta_parent / m;
        assert!(meta_parent < meta_idx);
        assert!(real_parent < node);

        (
            real_parent,
            DRSampleRanges {
                bucket: bucket_range,
                node: node_range,
            },
        )
    }

    /// Connect to `k` closest neighbors (see `KConnector`).
    fn connect_neighbors(&mut self, k: usize) {
        // FIXME: Let the algorithms initialize the slices instead of working
        //  only with vectors.
        // let parents = vec![vec![]; self.size()];
        // FIXME: How to set the capacity for the inner empty vector to `k`?

        debug_assert!(k > 0, format!("k {} is too small", k));

        // Check that the graph is big enough to accommodate k connections
        // at least in the center
        debug_assert!(
            self.size() - 2 * k > 0,
            format!(
                "the graph of size {} is too small for a k {}",
                self.size(),
                k
            )
        );

        for node in 0..self.size() {
            let smallest_parent = max(node as isize - k as isize, 0) as usize;
            self.parents.push((smallest_parent..node).collect());
        }
    }

    /// Uniformly set edges of exponentially increasing range.
    fn uniform(&mut self, m: usize, ner: usize) {
        self.parents = vec![vec![]; self.size()];

        for node in 1..self.size() {
            self.parents[node].push(node - 1);
        }

        let mut interval = ner;
        for _ in 0..m {
            let distance = interval + 1;
            for node in 0..self.size() {
                if distance > node {
                    continue;
                }
                self.parents[node].push(node - distance);
            }
            interval *= ner;
        }

        // At least the last node should have all the parents or we have chosen a poor
        // m/ner pair.
        assert_eq!(self.parents().last().unwrap().len(), m + 1);
    }

    // children_project returns the children edges denoted by this graph
    // instead of using the parent relationship.
    // If j = array[i][u] (for any u), then there is an edge (i -> j) in the graph.
    // Useful for the greedy attacks for example.
    pub fn children_project(&mut self) -> &Vec<Vec<usize>> {
        // compute only once
        if self.children.len() == 0 {
            let mut children = vec![vec![]; self.cap()];
            for (node, parents) in self.parents.iter().enumerate() {
                for &parent in parents.iter() {
                    children[parent].push(node);
                }
            }
            self.children = children;
        }
        return &self.children;
    }

    pub fn children(&self) -> &Vec<Vec<usize>> {
        if self.children.len() == 0 {
            panic!("called children() without children_project() first");
        }
        return &self.children;
    }

    /// Returns the number of edges
    pub fn count_edges(&self) -> usize {
        self.parents
            .iter()
            .fold(0, |acc, parents| acc + parents.len())
    }

    pub fn parents(&self) -> &Vec<Vec<Node>> {
        &self.parents
    }
    // FIXME: Remove this, at much return the parents of a single
    // node but do not allow complete access to the inner structure.

    pub fn for_each_edge<F>(&self, mut func: F)
    where
        F: FnMut(&Edge) -> (),
    {
        for (child, all_parents) in self.parents().iter().enumerate() {
            for &parent in all_parents.iter() {
                func(&Edge::new(parent, child));
                // FIXME: PERF: Maybe don't construct a new edge in every call.
            }
        }
    }
    // FIXME: Add an internal parent/edge iterator.

    // FIXME: Extend `F` definition to be able to return information (useful
    // to form new vectors from the original set of nodes).
    pub fn for_each_node<F>(&self, mut func: F)
    where
        F: FnMut(&Node) -> (),
    {
        for node in 0..self.size() {
            func(&node);
        }
    }

    pub fn cap(&self) -> usize {
        self.parents.len()
    }

    pub fn stats(&self) -> String {
        let mut parents = HashMap::new();
        let mut children = HashMap::new();
        self.for_each_edge(|edge| {
            let count = parents.entry(edge.child).or_insert(0);
            *count += 1;
            let count = children.entry(edge.parent).or_insert(0);
            *count += 1;
        });
        for i in 0..=self.degree() {
            parents.remove(&i);
        }
        let min_parent = parents.values().min().unwrap();
        let max_children = children.values().max().unwrap();
        format!(
            "graph stats: size={}, min parents={}, max children={}",
            self.size(),
            min_parent,
            max_children
        )
    }

    pub fn degree(&self) -> usize {
        match self.algo() {
            DRGAlgo::BucketSample => 2,
            DRGAlgo::MetaBucket(deg) => deg,
            DRGAlgo::KConnector(d) => d,
            DRGAlgo::UniformGraph {
                m,
                ner: _,
            } => m,
        }
    }

    /// buckets compute the different buckets Bi as defined in Alwen et al.
    /// to verify if they are all of relatively similar sizes. Since depth reducing
    /// set attacks are using the property that different buckets have different
    /// highly variable sizes, DRSample offers a core protection against these
    /// attacks by forcing the buckets to have ~ equal sizes.
    #[cfg(test)]
    fn buckets(&self) -> Vec<usize> {
        let log = (self.cap() as f32).log2().ceil() as usize;
        let mut ret = vec![0; log + 1];
        self.for_each_edge(|edge| {
            // dist = | u - v |
            let dist = (edge.child as i64 - edge.parent as i64).abs() as usize;
            // dist <= 2^î
            let i = (dist.next_power_of_two() as f32).log2().floor() as usize;
            ret[i] += 1;
        });
        return ret;
    }

    /// Convert the graph to a matrix where an `X` signals an edge
    /// connecting a parent row to a child column
    // FIXME: Decide the width based on the `size()` instead
    //  of hard-coding it here to a big number (3).
    pub fn to_str_matrix(&self) -> String {
        self.to_str_matrix_filtered(None)
    }

    pub fn to_str_matrix_filtered(&self, s: Option<&ExclusionSet>) -> String {
        let mut matrix = String::new();
        matrix += "\n";
        matrix += format!("{: >3}", "").as_str();
        for col in 0..self.size() {
            matrix += format!("{: >3}", col).as_str();
        }
        matrix += "\n";
        for row in 0..self.size() {
            matrix += format!("{: >3}", row).as_str();
            for col in 0..self.size() {
                matrix += format!(
                    "{: >3}",
                    if self.parents[col].contains(&row)
                        && (s.is_none() || !s.unwrap().contains(row) && !s.unwrap().contains(col))
                    {
                        "X"
                    } else {
                        ""
                    }
                )
                .as_str();
            }
            matrix += "\n";
        }
        matrix
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "G(cap:{}, drg: ", self.parents.len())?;
        match self.algo() {
            DRGAlgo::BucketSample => write!(f, "bucket, ")?,
            DRGAlgo::MetaBucket(d) => write!(f, "meta-bucket (degree {}), ", d)?,
            DRGAlgo::KConnector(k) => write!(f, "{}-connect, ", k)?,
            DRGAlgo::UniformGraph {
                m,
                ner,
            } => write!(
                f,
                "uniform-{}-m-{}-ner, ",
                m, ner
            )?,
        }
        write!(f, "parents: {:?}", self.parents)
    }
}

fn remove_duplicate<T: Hash + Eq>(elements: &mut Vec<T>) {
    let set: HashSet<_> = elements.drain(..).collect();
    elements.extend(set.into_iter());
}

#[cfg(test)]
pub mod tests {

    use super::*;
    use std::iter::FromIterator;

    pub static TEST_SEED: [u8; 32] = [1; 32];

    #[test]
    fn graph_new() {
        let size = 100;
        let g1 = Graph::new(size, TEST_SEED, DRGAlgo::BucketSample);
        assert_eq!(g1.parents.len(), 100); // no nodes generated yet
        assert_eq!(g1.parents.capacity(), size);
    }

    #[test]
    fn graph_bucket_sample() {
        let g1 = Graph::new(10, TEST_SEED, DRGAlgo::BucketSample);
        g1.parents.iter().enumerate().for_each(|(i, parents)| {
            if i == 0 {
                return;
            }
            // test there's at least a parent
            assert!(parents.len() >= 1 && parents.len() <= 2);
            // test there's at least the direct parent
            // == i since first cell is for node 1
            assert!(parents.iter().find(|x| **x == i - 1).is_some());
            // test the other parent is less
            if parents.len() == 2 {
                assert!(parents.iter().find(|x| **x < i).is_some());
            }
        });
    }

    #[test]
    fn graph_meta_sample() {
        let degree = 3;
        let g1 = Graph::new(10, TEST_SEED, DRGAlgo::MetaBucket(3));
        g1.parents.iter().enumerate().for_each(|(i, parents)| {
            if i == 0 {
                return;
            }
            // test there's at least a parent
            assert!(parents.len() >= 1 && parents.len() <= degree);
            // test there's at least the direct parent
            // == i since first cell is for node 1
            assert!(parents.iter().find(|x| **x == i - 1).is_some());
            // test all the other parents are less
            if parents.len() > 1 {
                assert_eq!(
                    parents
                        .iter()
                        .filter(|x| **x < (i - 1))
                        .collect::<Vec<&usize>>()
                        .len(),
                    parents.len() - 1
                );
            }
        });
    }

    #[test]
    fn graph_children_project() {
        // graph 1 - 5 nodes
        // 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4
        // 0 -> 2, 2 -> 4
        let p1 = vec![vec![], vec![0], vec![0, 1], vec![2], vec![2, 3]];
        let mut g1 = graph_from(p1);

        let children = g1.children_project();
        let exp = vec![vec![1, 2], vec![2], vec![3, 4], vec![4], vec![]];
        assert_eq!(children, &exp);
    }

    #[test]
    fn graph_remove() {
        // graph 1 - 5 nodes
        // 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4
        // 0 -> 2, 2 -> 4
        // Remove nodes 1 and 3
        // -> final graph 0 -> 2, 2 -> 4
        let p1 = vec![vec![], vec![0], vec![0, 1], vec![2], vec![2, 3]];
        let g1 = graph_from(p1);

        let nodes = ExclusionSet::from_nodes(&g1, vec![1, 3]);
        let g3 = g1.remove(&nodes);
        let expected = vec![vec![], vec![], vec![0], vec![], vec![2]];
        assert_eq!(g3.parents.len(), 5);
        assert_eq!(g3.parents, expected);
    }

    #[test]
    fn graph_depth() {
        let p1 = vec![vec![], vec![0], vec![1], vec![2], vec![3]];
        assert_eq!(graph_from(p1).depth(), 5);

        let p2 = vec![vec![], vec![], vec![0], vec![2], vec![2, 3], vec![3]];
        assert_eq!(graph_from(p2).depth(), 4);

        let size = 2048;
        let g3 = Graph::new(size, TEST_SEED, DRGAlgo::MetaBucket(3));
        assert_eq!(g3.depth(), size);
    }

    #[test]
    fn graph_count_edges() {
        let p1 = vec![vec![], vec![0], vec![1], vec![2], vec![3]];
        assert_eq!(graph_from(p1).count_edges(), 4);
        let p2 = vec![vec![], vec![], vec![0], vec![2], vec![2, 3], vec![3]];
        assert_eq!(graph_from(p2).count_edges(), 5);
    }

    #[test]
    fn graph_depth_exclude() {
        let p1 = vec![vec![], vec![0], vec![1], vec![2], vec![3]];
        let g1 = graph_from(p1);
        let s = ExclusionSet::from_nodes(&g1, vec![2]);
        assert_eq!(g1.depth_exclude(&s), 2);

        let g2 = Graph::new(17, TEST_SEED, DRGAlgo::MetaBucket(3));
        let s = ExclusionSet::from_nodes(&g2, vec![2, 8, 15, 5, 10]);
        let depthex = g2.depth_exclude(&s);
        assert!(depthex < (g2.cap() - s.size()));
        let g3 = g2.remove(&s);
        assert_eq!(g3.depth(), depthex);

        let size = (2 as usize).pow(10);
        let g3 = Graph::new(size, TEST_SEED, DRGAlgo::MetaBucket(3));
        assert!(g3.depth() <= size);
        let ssize = 2 ^ 6;
        let mut rng = ChaChaRng::from_seed(TEST_SEED);
        let mut sv = ExclusionSet::new(&g3);
        for _ in 0..ssize {
            sv.insert(rng.gen_range(0, size));
        }
        assert!(g3.depth_exclude(&sv) < size);
    }

    #[test]
    fn graph_depth_exclude_edges() {
        // 0->1->-2->3->4->5
        // 2->4
        // we remove 2->4 and 4->5
        // so max depth is 0->1->2->3->4 = 4 instead of 5
        let p2 = vec![vec![], vec![0], vec![1], vec![2], vec![2, 3], vec![4]];
        assert_eq!(graph_from(p2.clone()).depth(), 6);
        let edges = HashSet::from_iter(vec![Edge::new(2, 4), Edge::new(4, 5)]);
        assert_eq!(graph_from(p2).depth_exclude_edges(&edges), 4);
    }

    #[test]
    /// This test is testing the distribution of the edges of both bucket sample and
    /// drsample. As indicated in Alwen et al., when separating the edges into different
    /// buckets depending on the distance between the parent node and the child node,
    /// the buckets should have relatively same size. One sample is shown below:
    /// ```
    ///     drsample: size 32768
    ///             -> mean 2251.5, std_dev 112.204796 -> [1914.8856,2588.1143]
    ///             -> [32767, 3151, 2095, 2195, 2203, 2260, 2230, 2350, 2416, 2400, 237
    ///     7, 2286, 2085, 2121, 1881, 716]
    ///     drsample: size 32768
    ///             -> mean 4231.9165, std_dev 180.83716 -> [3689.405,4774.4277]
    ///             -> [32767, 3762, 4033, 4082, 4187, 4248, 4370, 4441, 4530, 4403, 426
    ///     2, 4153, 4216, 3858, 3544, 1275]
    ///
    /// ```
    /// We can see the first entry is always the same since these are the direct edges.
    /// The last entries have lower values since only a small portion of the nodes can
    /// have such a distance (2^9 <= dist(u,v) <= 2^10) with their parent, only the one
    /// that are at least at the index 2^9 !
    /// We can see for the rest of the buckets they are equivalently reqpresented, so the
    /// property is satisfied.
    /// NOTE: the test only start from the 3rd item and up to the third to last item in the
    /// bucket slices. The reason is the first 2 buckets have much higher number because
    /// the first nodes have a high pr. of falling into that buckets (no other choices).
    /// For the last buckets, see explanation before.
    fn graph_buckets() {
        let size = (2 as usize).pow(15);

        let test_dist = |g: &Graph| {
            let buckets = g.buckets();
            let normed = &buckets[2..buckets.len() - 2];
            let std = std_deviation(normed).unwrap();
            let mean = mean(normed).unwrap();
            let max = mean + 3.0 * std;
            let min = mean - 3.0 * std;
            println!(
                "drsample: size {}\n\t-> mean {}, std_dev {} -> [{},{}]\n\t-> {:?}",
                size, mean, std, min, max, buckets
            );
            assert_eq!(
                0,
                normed
                    .iter()
                    // three sigma rule
                    .filter(|&&v| v as f32 >= max || v as f32 <= min)
                    .count()
            );
        };
        let drsample = Graph::new(size, TEST_SEED, DRGAlgo::BucketSample);
        test_dist(&drsample);
        let bucket = Graph::new(size, TEST_SEED, DRGAlgo::MetaBucket(3));
        test_dist(&bucket);
    }
    fn mean(data: &[usize]) -> Option<f32> {
        let sum = data.iter().sum::<usize>() as f32;
        let count = data.len();

        match count {
            positive if positive > 0 => Some(sum / count as f32),
            _ => None,
        }
    }

    fn std_deviation(data: &[usize]) -> Option<f32> {
        match (mean(data), data.len()) {
            (Some(data_mean), count) if count > 0 => {
                let variance = data
                    .iter()
                    .map(|value| {
                        let diff = data_mean - (*value as f32);

                        diff * diff
                    })
                    .sum::<f32>()
                    / count as f32;

                Some(variance.sqrt())
            }
            _ => None,
        }
    }
    pub fn graph_from(parents: Vec<Vec<Node>>) -> Graph {
        Graph {
            spec: GraphSpec {
                size: (&parents).len(),
                seed: TEST_SEED,
                algo: DRGAlgo::BucketSample,
            },
            // FIXME: Same as `remove`, we shouldn't be creating graphs from
            //  parents (copying parents is almost like copying the entire
            //  graph for that matter).
            parents: parents,
            children: vec![],
        }
    }

    /// Fake RNG used only to generate values for `gen_range` taken from
    /// an internal pre-populated vector.
    // FIXME: There's probably a cleaner way without the references and lifetimes.
    struct FakeRNG<'a> {
        iter: &'a mut dyn Iterator<Item = &'a usize>,
    }
    impl UniformSampling for FakeRNG<'_> {
        fn sample_range(&mut self, range: &UniformSampleRange) -> usize {
            let value = *self.iter.next().expect("run out of numbers");
            let sample = (value % (range.high - range.low)) + range.low;
            assert!(sample >= range.low);
            assert!(sample < range.high);
            sample
        }
    }

    // Test the ranges over which BucketSample samples the bucket
    // and the node.
    // FIXME: The current numbers were just added after the fact to
    //  match the observed output. We should reason to check if they
    //  are correct and what should be expected in general.
    #[test]
    fn drsample_distributions_ranges() {
        let v = vec![1, 2, 3, 4, 5, 6];
        let i = v.iter();
        let mut rng = FakeRNG {
            iter: &mut i.cycle(),
        };

        let node = 10;
        let m = 5;

        let (_, ranges) = Graph::sample_parent_node(node, m, &mut rng);
        assert_eq!(ranges.bucket, UniformSampleRange { low: 1, high: 7 });
        assert_eq!(ranges.node, UniformSampleRange { low: 2, high: 5 });

        let (_, ranges) = Graph::sample_parent_node(node, m, &mut rng);
        assert_eq!(ranges.node, UniformSampleRange { low: 8, high: 17 });

        let (_, ranges) = Graph::sample_parent_node(node, m, &mut rng);
        assert_eq!(ranges.node, UniformSampleRange { low: 25, high: 51 });
    }
}
