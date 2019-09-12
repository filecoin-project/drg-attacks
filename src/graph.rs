use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::HashSet;
use std::hash::Hash;

use std::fmt;

// Graph holds the parameters and the edges of the graph. This is a special type
// of graph that has a *proper* labelling: for each edge (i,j), we have i < j.
#[derive(Debug)]
pub struct Graph {
    // parents holds all parents relationships of all nodes.
    // If j = parents[i][u] (for any u), then there is an edge (j -> i),
    // i.e. j is the parent of i.
    // The capacity of the graph is the size of the vector - Some nodes may be
    // absent when counted, i.e. node i may not have any parent and may not be
    // the parent of any other node. In that case, it is not included in the graph G
    parents: Vec<Vec<Node>>,
    // FIXME: Use slices, after construction this doesn't change.

    size: usize,

    seed: [u8; 32],
    algo: DRGAlgo,
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
        debug_assert!(parent < child);
        Edge {
            parent,
            child,
        }
    }
}

// DRGAlgo represents which algorithm can be used to create the edges so a Graph is
// a Depth Robust Graph
#[derive(Debug, Copy, Clone)]
pub enum DRGAlgo {
    // BucketSample is the regular bucket sampling algorithm with degree 2
    BucketSample,
    // MetaBucket is the meta-graph construction with a specified degree
    MetaBucket(usize),
}

impl Graph {
    // new returns a new graph instance from the given parameters.
    // The graph's edges are not generated yet, call fill_drg to compute the edges.
    pub fn new(size: usize, seed: [u8; 32], algo: DRGAlgo) -> Graph {
        let mut g = Graph {
            algo,
            seed,
            parents: Vec::with_capacity(size),
            size,
            children: vec![],
        };
        match g.algo {
            DRGAlgo::BucketSample => g.bucket_sample(),
            DRGAlgo::MetaBucket(degree) => g.meta_bucket(degree),
        }
        g
    }

    /// Number of nodes in the graph.
    // FIXME: Standardize size usage, don't access length or capacity of inner structures.
    pub fn size(&self) -> usize {
        self.size
    }

    // depth_exclude returns the depth of the graph when excluding the given
    // set of nodes
    pub fn depth_exclude(&self, set: &HashSet<usize>) -> usize {
        self.parents
            .iter()
            .enumerate()
            .fold(Vec::new(), |mut acc, (i, parents)| {
                if set.contains(&i) {
                    // an excluded node has length 0
                    acc.push(0);
                    return acc;
                }
                match parents
                    .iter()
                    // dont take parent's length if contained in set
                    .filter(|&p| !set.contains(p))
                    .map(|&p| acc[p] + 1)
                    .max()
                {
                    // need the match because there might not be any values
                    Some(depth) => acc.push(depth),
                    None => acc.push(0),
                }
                acc
            })
            .into_iter()
            .max()
            .unwrap()
    }

    // depth returns the longest depth found in the graph
    pub fn depth(&self) -> usize {
        self.parents
            .iter()
            .fold(Vec::new(), |mut acc, parents| {
                // take the depth of each parents + 1 then take the max of it
                match parents.iter().map(|&p| acc[p] + 1).max() {
                    Some(depth) => acc.push(depth),
                    None => acc.push(0),
                };
                acc
            })
            .into_iter()
            .max()
            .unwrap()
    }

    // remove returns a new graph with the specified nodes removed
    // TODO slow path checking in O(n) - consider using bitset for nodes
    pub fn remove(&self, nodes: &HashSet<usize>) -> Graph {
        let mut out = Vec::with_capacity(self.parents.len());
        for i in 0..self.parents.len() {
            let parents = self.parents.get(i).unwrap();
            let new_parents = if nodes.contains(&i) {
                // no parent for a deleted node
                Vec::new()
            } else {
                // only take parents which are not in the list of nodes
                parents
                    .into_iter()
                    .filter(|&parent| !nodes.contains(parent))
                    .map(|&p| p)
                    .collect::<Vec<usize>>()
            };
            out.push(new_parents);
        }

        Graph {
            size: (&out).len(),
            parents: out,
            algo: self.algo,
            seed: self.seed,
            children: vec![],
        }
    }

    // Implementation of the first algorithm BucketSample on page 22 of the
    // porep paper : https://web.stanford.edu/~bfisch/porep_short.pdf
    // It produces a degree-2 graph which is asymptotically depth-robust.
    fn bucket_sample(&mut self) {
        let mut rng = self.rng();
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
                    let max_bucket = (node as f32).log2().floor() as usize;
                    let i: usize = rng.gen_range(1, max_bucket + 1);
                    // get a node from that bucket, i.e. from [2^i-1, 2^i[
                    // exclusif because otherwise a parent can be the same
                    // as its child
                    let min = 1 << (i - 1);
                    let max = 1 << i;
                    let random_parent = rng.gen_range(min, max);
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
    fn meta_bucket(&mut self, degree: usize) {
        let mut rng = self.rng();
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
                    for _ in 0..m {
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
                        let i: usize = rng.gen_range(1, max_bucket + 1);
                        // choose parent in range [2^(i-1), 2^i[
                        let min = 1 << (i - 1);
                        // min to avoid choosing a node which is higher than
                        // the meta_idx - can happen since we can choose one
                        // in the same bucket!
                        let max = std::cmp::min(meta_idx, 1 << i);
                        assert!(max <= meta_idx);
                        let meta_parent = meta_idx - rng.gen_range(min, max);
                        let real_parent = meta_parent / degree;
                        assert!(meta_parent < meta_idx);
                        assert!(real_parent < node);
                        parents.push(real_parent);
                    }
                }
            }
            // filtering duplicate parents
            remove_duplicate(&mut parents);
            self.parents.push(parents);
        }
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

    fn rng(&self) -> ChaCha20Rng {
        ChaCha20Rng::from_seed(self.seed)
    }

    pub fn parents(&self) -> &Vec<Vec<Node>> {
        &self.parents
    }
    // FIXME: Remove this, at much return the parents of a single
    // node but do not allow complete access to the inner structure.

    pub fn for_each_edge<F>(&self, mut func: F)
    where F: FnMut(&Edge) -> () {
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
    where F: FnMut(&Node) -> () {
        for node in 0..self.size() {
            func(&node);
        }
    }

    pub fn cap(&self) -> usize {
        self.parents.len()
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "G(cap:{}, drg: ", self.parents.len())?;
        match self.algo {
            DRGAlgo::BucketSample => write!(f, "bucket, ")?,
            DRGAlgo::MetaBucket(d) => write!(f, "meta-bucket (degree {}), ", d)?,
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

    static TEST_SEED: [u8; 32] = [1; 32];

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

        let nodes = HashSet::from_iter(vec![1, 3].iter().cloned());
        let g3 = g1.remove(&nodes);
        let expected = vec![vec![], vec![], vec![0], vec![], vec![2]];
        assert_eq!(g3.parents.len(), 5);
        assert_eq!(g3.parents, expected);
    }

    #[test]
    fn graph_depth() {
        let p1 = vec![vec![], vec![0], vec![1], vec![2], vec![3]];
        assert_eq!(graph_from(p1).depth(), 4);

        let p2 = vec![vec![], vec![], vec![0], vec![2], vec![2, 3], vec![3]];
        assert_eq!(graph_from(p2).depth(), 3);
    }

    #[test]
    fn graph_depth_exclude() {
        let p1 = vec![vec![], vec![0], vec![1], vec![2], vec![3]];
        let g1 = graph_from(p1);
        let s = HashSet::from_iter(vec![2]);
        assert_eq!(g1.depth_exclude(&s), 1);

        let g2 = Graph::new(17, TEST_SEED, DRGAlgo::MetaBucket(3));
        let s = HashSet::from_iter(vec![2, 8, 15, 5, 10]);
        let depthex = g2.depth_exclude(&s);
        assert!(depthex < (g2.cap() - s.len()));
        let g3 = g2.remove(&s);
        assert_eq!(g3.depth(), depthex);
    }

    pub fn graph_from(parents: Vec<Vec<Node>>) -> Graph {
        Graph {
            size: (&parents).len(),
            parents: parents,
            seed: TEST_SEED,
            algo: DRGAlgo::BucketSample,
            children: vec![],
        }
    }
}
